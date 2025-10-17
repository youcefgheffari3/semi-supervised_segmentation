import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50
from voc_dataset import VOCDataset
from voc_pseudo_dataset import VOCPseudoDataset

def main():
    # --------------------------
    # Paths
    # --------------------------
    root_labeled = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_train_val"
    root_unlabeled = r"C:/Users/Gheffari Youcef/Downloads/semi-supervised_segmentation_approach-master/datasets/VOC2012_test"
    pseudo_dir = "pseudo_labels"

    # --------------------------
    # Datasets
    # --------------------------
    print("📂 Loading labeled and pseudo-labeled datasets...")

    labeled = VOCDataset(root_labeled, split="train", target_size=(256, 256))
    pseudo = VOCPseudoDataset(root_unlabeled, mask_dir=pseudo_dir, target_size=(256, 256))

    # Merge labeled + pseudo-labeled
    train_dataset = ConcatDataset([labeled, pseudo])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"✅ Total training samples: {len(train_dataset)} (Labeled: {len(labeled)}, Pseudo: {len(pseudo)})")

    # --------------------------
    # Model
    # --------------------------
    print("🧠 Initializing DeepLabV3-ResNet50 student model...")

    num_classes = 21
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --------------------------
    # Loss & Optimizer
    # --------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # --------------------------
    # Training Loop
    # --------------------------
    num_epochs = 20
    print(f"🚀 Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] finished, Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_name = f"deeplab_student_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"💾 Saved checkpoint: {checkpoint_name}")

    # --------------------------
    # Save final model
    # --------------------------
    torch.save(model.state_dict(), "deeplab_student_full.pth")
    print("🎉 Training complete! Student model saved as deeplab_student_full.pth")


if __name__ == "__main__":
    main()
