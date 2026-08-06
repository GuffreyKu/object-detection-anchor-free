"""Loopback check for train_remote.py: does a batch survive the RPC round trip?

    python test_remote_loader.py
"""
import multiprocessing as mp

import torch
import torch.distributed.rpc as rpc

import train_remote as tr

PORT = 29517
BATCHES = 3


def fake_batch(i):
    stacked = [torch.rand(2, 3, 8, 8), torch.rand(2, 4, 4, 5), torch.rand(2, 4, 4, 2),
               torch.rand(2, 4, 4, 2), torch.rand(2, 4, 4)]
    return (*stacked, [torch.tensor([[1., 2., 3., 4.]]) + i] * 2, [torch.tensor([0])] * 2)


def opts():
    return rpc.TensorPipeRpcBackendOptions(init_method=f"tcp://127.0.0.1:{PORT}", rpc_timeout=120)


def data_side():
    tr._loaders["train"] = [fake_batch(i) for i in range(BATCHES)]
    rpc.init_rpc(tr.DATA, rank=1, world_size=2, rpc_backend_options=opts())
    rpc.shutdown()


def gpu_side(q):
    rpc.init_rpc(tr.GPU, rank=0, world_size=2, rpc_backend_options=opts())
    try:
        loader = tr.RemoteLoader("train")
        got = list(loader)
        assert len(loader) == BATCHES, len(loader)
        assert len(got) == BATCHES, len(got)
        # two epochs: __iter__ must reset the far-side iterator, not run dry
        assert len(list(loader)) == BATCHES
        image, hm, wh, off, mask, boxes, labels = got[1]
        assert image.shape == (2, 3, 8, 8) and image.dtype == torch.float32
        assert torch.allclose(boxes[0], torch.tensor([[2., 3., 4., 5.]]))
        assert labels[0].tolist() == [0]
        q.put("ok")
    except BaseException as e:
        q.put(f"{type(e).__name__}: {e}")
    finally:
        rpc.shutdown()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=gpu_side, args=(q,)), ctx.Process(target=data_side)]
    [p.start() for p in procs]
    [p.join(180) for p in procs]
    result = q.get(timeout=5)
    assert result == "ok", result
    print("ok")
