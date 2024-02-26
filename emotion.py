import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from repvgg import create_RepVGG_A0 as create

# Load model
model = create(deploy=True)

# Define emotions
EMOTIONS = ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")

# Initialize device variable
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(device):
    """
    Initialize the emotion detection model.

    Args:
        device (torch.device): The device to load the model onto.
    """
    global dev
    dev = device
    model.to(device)
    model.load_state_dict(torch.load("weights/repvgg.pth"))

    # Save to eval
    cudnn.benchmark = True
    model.eval()

def detect_emotion(images, conf=True):
    """
    Detect emotions in a list of images.

    Args:
        images (list of numpy arrays): List of images represented as NumPy arrays.
        conf (bool): Whether to include confidence values in the result.

    Returns:
        list: List of emotions detected in the images.
    """
    with torch.no_grad():
        # Normalize and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        x = torch.stack([
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])(Image.fromarray(image)) for image in images
        ])

        # Feed through the model
        y = model(x.to(dev))
        result = []

        for i in range(y.size()[0]):
            # Get emotion index
            emotion_index = torch.argmax(y[i]).item()
            # Get emotion label
            emotion_label = EMOTIONS[emotion_index]

            if conf:
                confidence = 100 * y[i][emotion_index].item()
                result.append([f"{emotion_label} ({confidence:.1f}%)", emotion_index])
            else:
                result.append([emotion_label, emotion_index])

    return result
