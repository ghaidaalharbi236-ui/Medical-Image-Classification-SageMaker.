{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "1fbf74bd-5281-42df-8343-34b28cf0d846",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting hpo.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile hpo.py\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchvision\n",
    "from torchvision import datasets, models, transforms\n",
    "from PIL import Image\n",
    "import os\n",
    "import argparse\n",
    "\n",
    "def is_valid_image(path):\n",
    "    try:\n",
    "        with Image.open(path) as img:\n",
    "            img.verify()\n",
    "        return True\n",
    "    except:\n",
    "        return False\n",
    "\n",
    "def train(args):\n",
    "    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])\n",
    "    dataset = datasets.ImageFolder(args.data_dir, transform=transform)\n",
    "    dataset.samples = [s for s in dataset.samples if is_valid_image(s[0])]\n",
    "    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)\n",
    "    \n",
    "    model = models.resnet18(pretrained=True)\n",
    "    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))\n",
    "    \n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.Adam(model.parameters(), lr=args.lr)\n",
    "    \n",
    "    model.train()\n",
    "    for epoch in range(1, args.epochs + 1):\n",
    "        running_loss = 0.0\n",
    "        for data, target in loader:\n",
    "            optimizer.zero_grad()\n",
    "            output = model(data)\n",
    "            loss = criterion(output, target)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            running_loss += loss.item()\n",
    "        print(f\"Test set: Average loss: {running_loss / len(loader):.4f}\")\n",
    "\n",
    "    torch.save(model.state_dict(), os.path.join(args.model_dir, \"model.pth\"))\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    parser = argparse.ArgumentParser()\n",
    "    parser.add_argument('--lr', type=float, default=0.001)\n",
    "    parser.add_argument('--batch-size', type=int, default=4)\n",
    "    parser.add_argument('--epochs', type=int, default=1)\n",
    "    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))\n",
    "    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))\n",
    "    args, _ = parser.parse_known_args()\n",
    "    train(args)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "d1c85a13-3ba2-46c0-8d20-9eadf4a0049c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/sagemaker-user\n",
      "covid-chestxray-dataset-master/  data/\tmedical_data/  user-default-efs/\n"
     ]
    }
   ],
   "source": [
    "!pwd\n",
    "!ls -d */"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d6cf0601-fe3e-400c-8498-1f7e1bd664f0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
