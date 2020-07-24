# TO DO LIST / RESULTS
    To try to learn how to classify tweet responses into the following categories
    0. Explicit Denial
    1. Implicit Denial
    2. Implicit Support
    3. Explicit Support
    4. Comment
    5. Queries
    
    Will try to use BERT to do the task    
    
## **Task 0: Preprocessing Data**
    [x] Visualize category labels density
    [x] Filter the NaNs in either parent tweet or child tweet
    [x] Tokenize tweet pairs
    [x] Convert labels into numbers
    [x] Split the dataset into training and test sets
    [x] Ensure test dataset label density is similar to main dataset's
    [x] Save the tokenized tweets into binaries    

### **Data:**
    Here's how the data looks like.

![Tweet labels](./data/label_density.png)
        
    There are 5221 tweet pairs in the raw data. 
    However, there are a lot of deleted tweets.
    There are 4271 tweet pairs in the cleaned data.
        ~90% training set (3844)
        ~10% test set (427)
    Always fix the test set as the last 427 samples

    After splitting into training and test sets, the densities still look OK.
    
![Train vs test sets](./data/test_train_density.png)

## **Task 1: Try to classify using BERT**
    [ ] Change the F1 metric to macro F1 score
    [ ] Build a simple BERT first
    [ ] Improve old code for saving training steps
    [ ] Save intermediate models

## **Task 2: Try to classify using XLNet**

## IGNORE EVERYTHING BELOW FOR NOW

    After flattening, learn classification pairwise, with 6 different categories.

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
    NOT DONE YET
    
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
    

# Concepts/tools used for this exercise
    ~~pytorch: ~~
        ~~how to build NNs~~
        ~~how to train and use NNs~~
        ~~huggingface transformers library~~
    ~~CUDA stuff: ~~
        ~~moving things to GPU only when needed~~
        ~~deleting references to objects no longer needed~~
        ~~release memory by calling cuda.emptying_cache()~~
        ~~if all else fails, backup models, data, then reboot python kernel~~
    ~~General stuff:~~
        ~~Practice proper file handling to prevent overwrite accidents~~
        ~~Saving and caching tokenizer outputs. Tokenizing the entire dataset is damn slow. ~3.5hr~~
    