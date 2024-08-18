import torch

model = torch.load('savemodel/model_trace.pt', map_location="cpu")

model.eval()
image_size = (512, 512)

if __name__ == '__main__':
    input_x = torch.rand(1, 1, image_size[1], image_size[0])

    torch.onnx.export(model,
                        input_x,
                        input_names=["input"],
                        output_names=["output"],
                        f="savemodel/model.onnx")