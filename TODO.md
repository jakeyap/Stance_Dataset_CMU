# **TO DO LIST / RESULTS **
    To try to learn how to classify tweet responses into the following categories
    0. Explicit Denial
    1. Comment
    2. Explicit Support
    3. Implicit Denial
    4. Queries
    5. Implicit Support
    
    Will try to use BERT to do the task    

## **Task 0: Preprocessing**
    [x] Visualize category labels density
    [x] Filter the NaNs in either parent tweet or child tweet
    [x] Draw a histogram of token lengths for posts to truncate appropriately
    [ ] Tokenize tweet pairs
    [ ] Convert labels into numbers
    [ ] Split the dataset into training and test sets
    [ ] Ensure test dataset label density is similar to main dataset's
    [ ] Save the tokenized tweets into binaries    
    [ ] Change the metric to macro F1 score

## **Task 1: Try to classify using BERT**


### Method A
    After flattening, learn classification pairwise, with 100 different categories.
    
### **Data:**
    Here's how the data looks like.

![Comment labels](./data/label_density.png)
        
    There are 5221 tweet pairs in the raw data. 
    However, there are a lot of deleted tweets.
    There are 4271 tweet pairs in the cleaned data.
        ~90% training set (3844)
        ~10% test set (427)
    Always fix the test set as the last 427 samples
    
## IGNORE EVERYTHING BELOW FOR NOW

### **Models:**
    
    
### ModelA1
    BERT ==> Dropout1 10% ==> Linear1 ==> RELU1
         ==> Dropout2 10% ==> Linear2 ==> RELU2 ==> Linear3
    Loss: Cross Entropy Loss, flat weights
    
### ModelA2
    BERT ==> Linear
    Loss: Cross Entropy Loss, inverse weights to label occurence
        add 1k to all counts
        then divide the sum by each element
        
### ModelA3
    BERT ==> Linear
    Loss: Cross Entropy Loss, inverse weights to label occurence
        add 1k to all counts
        then divide the sum by each element
        then divide by the biggest number to normalize to 1 or lower
    
### ModelB1 
    BERT ==> Dropout1 10% ==> Linear1 ==> RELU1 ==> Dropout2 10% ==> Linear2
            parent_label  ==>
    Loss: Cross Entropy Loss, flat weights
    
### ModelB2
    Same as model B1
    TODO: Loss: weighted loss. 

### ModelC1
    Same as model B1
    
### ModelC2
    Same as model B2
    TODO

### **Training algo:**
    SGD
        Learning rate = 0.001
        Momentum = 0.5
        Minibatch size = 40

### **Hardware used:**
    GPU: RTX 2080 Super (8Gb RAM)
    CPU: Ryzen 3900 (12 cores 24 threads)
    1 epoch takes ~40 min for modelA
    Peak GPU RAM usage is ~7/8 Gb. Dont use maximum to give some buffer

### **Results:**

### ModelA1
    Emphirically, after 6-7 epochs, overfitting kicks in. 
    
![overfit](./results/MODELA/modelA1_overfitting.png)

    Stop training at 6 epochs then. The ball park accuracy is 53%-56%.

![loss](./results/MODELA/modelA1_losses_6epochs.png)

![predicted labels](./results/MODELA/modelA1_predicted_labels_6_epochs.png)

    For comparison, here's the real label density for the test set
![testset labels](./results/testset_labels.png)
    
### ModelA2
    Not meaningful. Didn't learn after 10 epochs
    
### ModelA3
    Peak accuracy of 47% after 7 training epochs. 

![ModelA3 loss](./results/MODELA/modelA3_losses.png)

    
### ModelB1
    See ModelC1. Trained using true parent's label.
    Accuracy is 78-80%. F1 score is 0.79-0.80.
    
### ModelB2
    TODO
    
### ModelC1
    The peak accuracy after 10 epochs is 80-82%. F1 score of 0.80-0.83.
![ModelC1 loss](./results/MODELB/modelC1_losses_10epochs.png)

### ModelC1
    TODO
    
### **Remarks:**
    How to deal with abbreviations? Some examples
    - NRA
    - ROFL
    - LOL
    
    Comment pairs, heavily skewed towards the (question,answer) label, so the other types seem to get drowned out.
    In order to account for that, perhaps need to weigh the cost function, to decrease cost associated with (question,answer) label
    The 3 categories <humor>, <negative reaction>, <other>, are very under represented. Perhaps their cost need to be weighted upwards
    Perhaps increase the batch size and lower the tokenization length.
    Further handicaps
        -Looking at a comment pair with no context. How do you tell whether it is an announcement or elaboration?
        
    I tried to weigh the cost function (see ModelA2 and ModelA3 above for details). ModelA2 didnt learn at all. ModelA3 somewhat works OK, with a peak accuracy of 47% after 7 training epochs. The losses are shown above in results section.
    
    
# Task 2: Maintain tree structure 
    Use PLAN model. Not sure how yet.


# Concepts/tools used for this exercise
    pytorch: 
        how to build NNs
        how to train and use NNs
        huggingface transformers library
    CUDA stuff: 
        moving things to GPU only when needed
        deleting references to objects no longer needed
        release memory by calling cuda.emptying_cache()
        if all else fails, backup models, data, then reboot python kernel
    General stuff:
        Practice proper file handling to prevent overwrite accidents
        Saving and caching tokenizer outputs. Tokenizing the entire dataset is damn slow. ~3.5hr
    