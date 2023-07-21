# AWS-image_classifier
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, We'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) dataset of 102 flower categories.

### After cloning the repository,follow the following steps to use the command line application.

Make sure you have all the required packages installed, such as numpy, matplotlib, seaborn, torch, torchvision, and PIL. If you don't have these packages, you can install them using pip:
```
pip install numpy matplotlib seaborn torch torchvision pillow
```
Open a terminal or command prompt and navigate to the directory where the train.py script is located and run the following command:
```
python train.py --data_dir /path/to/data --save_dir /path/to/save/checkpoint.pth --arch vgg19 --learning_rate 0.003 --hidden_units 250 --epochs 1 --gpu gpu
```
Here, you can modify the command-line arguments as per your requirement:

--data_dir: Path to the root directory containing the 'train,' 'valid,' and 'test' directories.

--save_dir: Path to save the checkpoint file (default is "./checkpoint.pth").

--arch: The architecture to use for the model (options: "vgg19," "densenet121," or "resnet152").

--learning_rate: Learning rate for the optimizer (default is 0.003).

--hidden_units: Number of hidden units in the classifier (default is 250).

--epochs: Number of training epochs (default is 1).

--gpu: Use "gpu" to enable GPU training or "cpu" for CPU training (default is "gpu").

### If you dont't want to train your own network, you can also find pretrained weights [here](https://drive.google.com/file/d/1FP4HTH5J8aVbbnztwTzm-HNSTS6f0KqT/view?usp=sharing).
Now ,for inferences :
Run the script with the desired arguments:
```
python predict.py --image path/to/your/image.jpg --checkpoint path/to/your/checkpoint.pth
```

--image: The path to the image you want to classify.

--checkpoint: The path to the saved checkpoint of the trained model.

--top_k: (Optional) The number of top predicted classes to display (default is 5).

--category_names: (Optional) The path to the JSON file that maps category labels to their corresponding flower names (default is cat_to_name.json).

--gpu: (Optional) Specify gpu to use the GPU for inference or cpu to use the CPU (default is gpu).
