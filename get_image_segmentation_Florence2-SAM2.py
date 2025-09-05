import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

FLORENCE2_MODEL_PATH = "/media/chen/data/all_models/Florence-2"

SAM2_CHECKPOINT = "/media/chen/workspace/workspace/traffic_light_auto_labeling/checkpoints/sam2.1_hq_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_PATH, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_PATH, trust_remote_code=True)
print("load florence-2 model successfully")
# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)



def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer


def caption_phrase_grounding_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    caption_task_prompt='<DETAILED_CAPTION>',
    output_dir="viz/"
):
    # assert caption_task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
    image = Image.open(image_path).convert("RGB")
    
    # image caption
    caption_results = run_florence2(caption_task_prompt, None, florence2_model, florence2_processor, image)
    text_input = caption_results[caption_task_prompt]
    print(f'Image caption for "{image_path}": ', text_input)
    
    # phrase grounding
    grounding_results = run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', text_input, florence2_model, florence2_processor, image)
    grounding_results = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
    # grounding_results = text_input
    # parse florence-2 detection results
    input_boxes = np.array(grounding_results["bboxes"])
    class_names = grounding_results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite(os.path.join(output_dir, f"{os.path.basename(image_path)}.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, f"{os.path.basename(image_path)}.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')

if __name__ == "__main__":
    root = "/media/chen/data/nusecnes/v1.0-mini/samples/CAM_FRONT"
    for image_path in os.listdir(root):
        image_path = os.path.join(root, image_path)
        caption_phrase_grounding_and_segmentation(
            florence2_model,
            florence2_processor,
            sam2_predictor,
            image_path,)