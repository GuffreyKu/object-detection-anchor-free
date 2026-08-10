"""Is the RPC link up? Run this before blaming train_remote.py.

    # GPU box first
    python check_remote.py --role gpu  --addr <gpu-ip>
    # then here
    python check_remote.py --role data --addr <gpu-ip>

Imports nothing from this repo, so it isolates the network from the dataloader,
and it prints a line per stage so you can see exactly where it stops.
"""
import argparse
import os
import socket
import time

import torch
import torch.distributed.rpc as rpc

GPU, DATA = "gpu", "data"


def echo(t):
    return f"{socket.gethostname()} received {t.numel() * t.element_size() / 1e6:.0f} MB"


def assert_port_free(addr, port):
    """rank 0 owns the rendezvous port, so a stale run still holding it is invisible:
    the peer connects to the zombie's store, completes the handshake, and then waits
    forever for a rank 0 that will never answer. Fail loudly instead."""
    s = socket.socket()  # no SO_REUSEADDR: we want this to fail if anyone is listening
    try:
        s.bind((addr, port))
    except OSError as e:
        raise SystemExit(
            f"cannot bind {addr}:{port} ({e})\n"
            f"A previous run is probably still holding it. Find and kill it:\n"
            f"    lsof -nP -iTCP:{port} -sTCP:LISTEN    # macOS\n"
            f"    ss -lptn 'sport = :{port}'            # linux")
    finally:
        s.close()


def net_setup(addr, iface=None):
    """Pin gloo and TensorPipe to one interface and report the address we will use.

    On a multi-homed box (VPN + LAN, which is the normal case for remote training)
    both pick an interface on their own and can land on one the peer cannot route
    to. Rendezvous still succeeds, because that connection is outbound; the reverse
    channel then hangs with no error. Must run before init_rpc.
    """
    if iface:
        os.environ["GLOO_SOCKET_IFNAME"] = iface
        os.environ["TP_SOCKET_IFNAME"] = iface

    # connect() on UDP sends nothing, it just asks the kernel which source address
    # it would route from.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((addr, 9))
        src = s.getsockname()[0]
    finally:
        s.close()

    host_ip = socket.gethostbyname(socket.gethostname())
    print(f"      route to {addr} leaves from {src}"
          + (f" (pinned to {iface})" if iface else ""), flush=True)
    if not iface and host_ip != src:
        print(f"      WARNING: hostname resolves to {host_ip}, not {src}. Multi-homed host:"
              f" gloo/TensorPipe may advertise the wrong one. Pass --iface <interface>.",
              flush=True)
    return src


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=[GPU, DATA], required=True)
    p.add_argument("--addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    p.add_argument("--mb", type=int, default=64, help="payload size for the throughput probe")
    p.add_argument("--iface", default=None,
                   help="network interface facing the peer, e.g. ppp0 / utun3 / eth0. "
                        "Needed on a multi-homed host - see net_setup().")
    args = p.parse_args()
    rank = 0 if args.role == GPU else 1
    net_setup(args.addr, args.iface)
    if rank == 0:
        assert_port_free(args.addr, args.port)

    if rank == 1:
        # Plain TCP first: this tells "master unreachable" apart from "rendezvous ok
        # but TensorPipe cannot open its own sockets back".
        print(f"[1/4] dialing tcp://{args.addr}:{args.port} ...", flush=True)
        try:
            socket.create_connection((args.addr, args.port), timeout=10).close()
            print("      reachable", flush=True)
        except OSError as e:
            print(f"      UNREACHABLE: {e}")
            print("      -> the GPU box is not listening there. Start --role gpu first,")
            print("         and make sure it binds an address it actually owns (a cloud VM")
            print("         cannot bind its public IP - use the private IP or a VPN address).")
            return

    opts = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{args.addr}:{args.port}", rpc_timeout=30)

    print(f"[2/4] init_rpc as {args.role} (rank {rank}) ...", flush=True)
    rpc.init_rpc(args.role, rank=rank, world_size=2, rpc_backend_options=opts)
    print(f"      rendezvous ok, this host is {socket.gethostname()}", flush=True)

    if rank == 0:
        try:
            print("[3/4] control ping ...", flush=True)
            print("     ", rpc.rpc_sync(DATA, echo, (torch.zeros(4),)), flush=True)

            n = args.mb * 1_000_000 // 4
            print(f"[4/4] pulling {args.mb} MB ...", flush=True)
            t0 = time.perf_counter()
            rpc.rpc_sync(DATA, echo, (torch.zeros(n),))
            dt = time.perf_counter() - t0
            print(f"      {dt:.1f}s -> {args.mb / dt:.1f} MB/s", flush=True)
            print(f"      a 512x512 batch of 32 is ~50 MB at fp16: {50 / (args.mb / dt):.1f}s each",
                  flush=True)
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {e}")
            print("      -> rendezvous worked but the data channel did not. TensorPipe opens")
            print("         its own sockets in BOTH directions and advertises each host's own")
            print("         interface address, so an ssh -L tunnel or NAT breaks it. Put both")
            print("         machines on one LAN or VPN (Tailscale/WireGuard).")
    else:
        print("[3/4] serving, waiting for the GPU box to finish ...", flush=True)

    rpc.shutdown()
    print("[4/4] clean shutdown", flush=True)


if __name__ == "__main__":
    main()
