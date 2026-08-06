import torch

image_size = (512, 512)

if __name__ == '__main__':
    # model_trace.pt is written by torch.jit.save, so it needs torch.jit.load.
    model = torch.jit.load('savemodel/model_trace.pt', map_location="cpu")
    model.eval()

    input_x = torch.rand(1, 3, image_size[1], image_size[0])

    torch.onnx.export(model,
                        input_x,
                        f="savemodel/model.onnx",
                        input_names=["input"],
                        output_names=["hm", "wh", "offset"],
                        opset_version=17)
