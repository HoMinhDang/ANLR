import detect_char
import detect_plate
import argparse
import torch
import os


def save_plate_number(plate_number, cropped_path):
    output_dir = os.path.dirname(cropped_path)
    base_name = os.path.splitext(os.path.basename(cropped_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_char.txt")
    with open(output_path, "w") as f:
        f.write(plate_number)

    print(f"Saved plate number to: {output_path}")

def main(img_path, model_plate, model_char, device):
    """
        Pipeline for Automated License Plate Recognition
    """
    
    # init model
    plate_model = detect_plate.PlateDetector(weights=args.model_plate) 
    char_model = detect_char.CharDetector(weights_path=args.model_char, device=device)

    try:
        # Detect plates
        cropped_images, labels, cropped_paths = plate_model.detect_plate(img_path)
    except ValueError as e:
        print(f"Error in plate detection: {e}")
        return
    
    for img, label, path in zip(cropped_images, labels, cropped_paths):
        try:
            plate_number = char_model.detect_char(img, label)
            print(f"Detected plate number: {plate_number}")
            save_plate_number(plate_number, path)
        except ValueError as e:
            print(f"Error in character detection: {e}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated License Plate Recognition")
    parser.add_argument("--img", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model_plate", type=str, default="object.pt", help="Path to the plate detection model")
    parser.add_argument("--model_char", type=str, default="char_custom.pt", help="Path to the character detection model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    main(args.img, args.model_plate, args.model_char, args.device)

