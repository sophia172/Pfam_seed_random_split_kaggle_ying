The detailed python notebook can be found [here](https://github.com/sophia172/Pfam_seed_random_split_kaggle_ying/blob/main/Protein_Sequence_Classification.ipynb)
# Problem Setup

Classify sample of protein sequence to a domain family_accession label. Each sample is a protein sequence which has different length from about 20 to about 1900. There are nearly 18000 domain family_accession labels. 
So it is a multi-classification problem with large dataset and big feature space. 

# Data Structure

There are three sets of data: train, dev, and test. 
- The train set contains 1,086,741 samples. 
- The dev set contains 126,171 samples. 
- The test set contains 126,171 samples. I will not use it during the whole process.

Each sample contains information of:
```
sequence: HWLQMRDSMNTYNNMVNRCFATCIRSFQEKKVNAEEMDCTKRCVTKFVGYSQRVALRFAE 
family_accession: PF02953.15
sequence_name: C5K6N5_PERM5/28-87
aligned_sequence: ....HWLQMRDSMNTYNNMVNRCFATCI...........RS.F....QEKKVNAEE.....MDCT....KRCVTKFVGYSQRVALRFAE 
family_id: zf-Tim10_DDP
```

Description of fields:

- sequence: These are usually the input features to your model. Amino acid sequence for this domain.  
    There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite   
    uncommon: X, U, B, O, Z.
- family_accession: These are usually the labels for your model. Accession number in form PFxxxxx.y   
    (Pfam), where xxxxx is the family accession, and y is the version number.   
    Some values of y are greater than ten, and so 'y' has two digits.
- family_id: One word name for family.
- sequence_name: Sequence name, in the form "uniprot_accession_id/start_index-end_index".
- aligned_sequence: Contains a single sequence from the multiple sequence alignment (with the rest of the members of  the family in seed, with gaps retained.

**Input** : The input can be either *sequence* data or *aligned_sequence* data.
**Output**: The output label is *family_accession*

# Data Analysis

## Label (refer to Python notebook section 2.1)

### Check how many labels do train/dev/test set have.

- Check the label in one train file 
	The number of unique family id or unique *family_accession* are both 6176 in just one file. It means there are a big number of labels. unique *family_id* has the same number as *family_accession*. So I will only look at *family_accession*.

- *family_accession* has two parts *x* and *y*. Check the difference.
	```PFxxxx.y```
	There are 6176 unique value for the front part *x* and 33 unique values for the back part *y*.
	It means only the front part contributes to the labels. The back part probably mean something else. 

- Double check for all files to make sure my analysis is consistent in all files.
	Confirmed

#### Check the number of samples in each *family_accession*. (Python notebook section2.1.1)

- The sample numbers in each *family_accession* label are different. It could be either 1 or over 3500. 
- There are only 13071 labels in the dev set and 17929 labels in the train set. So there are a lot samples we do not need. But we cannot guarantee that the labels will not show up in the test set. So we can remove the ones with less training sample.
- After examination, I found out that every label which showed up in dev set has more than 7 training samples in the train set. So I remove the ones which have less than 8 training samples within the label. 
#### The train set filter process (Python notebook section 2.1.2)

- I kept 1,072,097 samples in the train set with 13912 labels.
#### Check on the back part of the label to see any difference  (Python notebook section 2.1.3)

-  TODO
## Sequence (Python notebook section 2.2)

### Check the number of uncommon amino acids in all smaples

- There are only 747 sequence containing uncommon amino acids.

### Check the length of sequence (Number of amino acids)

- The length ranges from about 20 up to 1972. 
- The unique amino acids are
	```'D', 'O', 'N', 'V', 'S', 'E', 'G', 'X', 'Z', 'F', 'M', 'Y', 'B', 'K', 'R', 'Q', 'C', 'A', 'I', 'W', 'P', 'U', 'H', 'T', 'L'```
- By comparing to published unique amino acids list. I used this numeric encoding dictionary *str_to_int* (in Python notebook section 2.3)

### Check whether the number of  unique amino acids in each sample has correlations with each other.

- Z, U, B, X, O are almost indepandent with the others. It fits the discription that they are uncommon.
- Z is strongly correlated with B, But due to the small occurrence, it does not mean anything.

### Check if they have any relationship with the front part of *family_accession*

- Nothing observed

### Check if they have any relationship with the back part of *family_accession*

- It looks like the back part of *family_accession* is related to the number of unique amino acid in each sample. 
- TODO - cluster samples based on the back part of *family_accession*. Run classification in each cluster 

## Aligned sequence (Python notebook section 2.3)

### Encode aligned sequence and check how they are related in each family

- Notice that the length of aligned sequence are the same within each *family_accession*
- Notice that some families have the same length of aligned sequence.
- Notice that the aligned sequence in the same family have similar pattern. 
![[aligned_sequence.jpg]]
### Classification strategy
- In the samples with the same *aligned_sequence* length, calculate the centroid for each acid position in one family. whichever family centroid is closer to the new sample will be the family label.


# Model development 1

## Use *aligned_sequence* as input (Python notebook section 3.1)

- Find the length of inputs.
- Each length contains 1 to many families.
- Save the length with corresponding families

## Develop model (Python notebook section 4.1)

- For one specific length, Find all the training set and corresponding families. 
- Calculate the centroid of the family samples. 
- Get a new sample for dev set with the same input length, calculate the norm between the new sample and each family centroid. 
- Assign the closest centroid as the predicted family label
- Result proven to be 100% accurate every time. 
- I did not explore the whole dataset. From the random 20 length I checked, the accuracy remains the same. 

# Model development 2
## Use sequence as input (Python notebook section 3.2)

Since protein sequence cannot always be aligned nicely and manually checked, It is worth to work with *sequence* as input. There are two feature extraction methods I am testing. One is general **Autoencoder** (Python notebook section 3.2.1). Another one is **UniRep** (Python notebook section 3.2.2). Either of these methods were probably tuned and tested. But I hope you get an idea of what I want to do. 

Both methods requires zero padding for the input as they have very different length and need to be fed to a model. Since some samples have only about 20 acids and some have over 1900 acids, they cannot all be padded to the same length. Hence, I am applying gridded padding to the input. 

For instance, If I have samples as:
```
input = 
   [[9,2,4,2], 
	[4,5], 
	[6,7,8,3,66,4,3]]
```

I want to pad them in length with steps of 5. It means I will pad them to either 5 or 10 or 15 and etc. 

So the padded input becomes
```
input = 
   [[9,2,4,2,0], 
	[4,5,0,0,0], 
	[6,7,8,3,66,4,3,0,0,0]]
```

The Input padding algorithm are written in Python notebook section 3.2.1. A similar method is embedded in online UniRep JAX package but might be slightly different. It is not shown in the Python notebook section 3.2.2 but used for the feature extraction there. 

## Feature analysis  (Python notebook section 3.2.3)

### Analysis for Autoencoder feature: 

- PC1 has range from 0 to 300 while PC2 is 10000 times smaller and PC3 is even worse. Potentially the PCA need to be whitened.
- The PCs forms connecting straight lines ==>
	It means that the data can be separate into different clusters. each cluster owns the data in this straight line. In each cluster, the data can be spearate linearly.
- The colour does not change smoothly for both front of the family and back of the family.

  

**Next step**:

- try DBscan to cluster samples; --> Proved not good clustering in the later analysis.
- Try decision tree.

  

### Analysis for UniRep feature:

- The PCA feature does not show obvious clustering from the observation ==> no unsupervised clustering.
- It does not suit few-shot learning unless it is expanded to higher dimension with the help of blessing of dimensionality. But I do not recommand that as 1900 is already a big dimension.

**Next step**:

- try some simple fast classifier

## Develop model (Python notebook section 4.2)

### Define metric (Python notebook section 4.2.1)

If I am using regression to solve this classification problem, I can use Mean Squared Error as the metrics.

### Tested some conventional ML method using extracted features to classify train set as a whole (Python notebook section 4.2.3.1)
- Neural network
- Decision Tree
### Tested some conventional ML methods using extracted features to cluster data first (Python notebook section 4.2.3.2)
- DBscan
- Spectral cluster
### Tested few-shot learning using extracted features to run binary classification (Python notebook section 4.2.3.3)

- The confusion matrix show that it performs all right but not 100% accurate

## Improvement (TODO)

- Few shot learning but with clustering using the back of *family_accession* as shown in the end of Python notebook section 2.2
- Improve on the feature extraction. The current feature extraction is not good at all. 


