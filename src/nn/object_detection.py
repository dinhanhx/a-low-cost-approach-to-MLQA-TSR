import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


class ObjectDetection:
    def __init__(self) -> None:
        self.model_name = "google/owlv2-large-patch14-ensemble"
        self.labels = [
            "traffic sign",
            "rectangle sign",
            "triangle sign",
            "square sign",
            "circle sign",
            "octagon sign",
            "stop sign",
        ]
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
        self.processor = Owlv2Processor.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self.model.eval().cuda()

    def infer_batch(self, images: list[Image.Image], threshold: float = 0.3):
        image_sizes = [i.size[::-1] for i in images]
        model_inputs = self.processor(
            images=images,
            text=[self.labels] * len(images),
            return_tensors="pt",
        )
        model_inputs = model_inputs.to("cuda")

        with torch.no_grad():
            model_outputs = self.model(**model_inputs)

        results = self.processor.post_process_object_detection(
            outputs=model_outputs,
            target_sizes=torch.Tensor(image_sizes),
            threshold=threshold,
        )
        for i in results:
            i["labels"] = [self.labels[i] for i in i["labels"]]
            i["boxes"] = i["boxes"].cpu().numpy().tolist()  # x_min, y_min, x_max, y_max
            i["scores"] = i["scores"].cpu().numpy().tolist()

        torch.cuda.empty_cache()
        return results
