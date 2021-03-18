
# Oh gosh! Is that a bitcoin heist?


## Introduction

Ransomware is a type of malicious software that locks up a user’s computer system until a ransom is paid, a virtual heist if you will. Payments are often demanded through Bitcoin, an anonymous currency. 

But not all is lost! Bitcoin transactions are public and permanently stored – this leaves plenty of data for a machine learning intervention.

 * Queue hopeful music *

Hence, my research prompt: can I use machine learning to tell whether a ransomware was behind a bitcoin transaction?


## About the data

I used this dataset contributed to the UCI Machine Learning Repository.

[https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset)

The data contains bitcoin transactions above a certain threshold amount collected over a 24 hour period. 

I’ve taken some steps to simplify the data I will working with:



*   Turned multiclass target labels into binary ones: white or ransomware. 
*   Sampled only 20% of the original data, amounting to 570k instances. (Otherwise training models would choke my computer). 


## EDA

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


Right off the bat, I saw something less than desirable. Feature variables were all minimally correlated with the target variable, _label_. Nevermind that, the project’s gotta go on.

Additionally, the dataset was quite imbalanced, offering a great opportunity to experiment with various sampling techniques. 


<table>
  <tr>
   <td>
   </td>
   <td>Count
   </td>
   <td>Proportion
   </td>
  </tr>
  <tr>
   <td>White (encoding=0)
   </td>
   <td>575,056
   </td>
   <td>98.6%
   </td>
  </tr>
  <tr>
   <td>Ransomware (encoding=1)
   </td>
   <td>8,282
   </td>
   <td>1.4%
   </td>
  </tr>
</table>


Lastly, there were quite a few right skewed features, but a quick log transformation did the job.


## Preprocessing

The preprocessing pipeline (built with scikit-learn)  was kept nice and simple, as the data was already quite clean. Here are the steps:



1. Data imputation. Missing categorical values were replaced with the string ‘Missing’ , and missing numeric ones were replaced with an anomalously larget number -99999. 

Skills showcased 

Scikit-learn Pipeline

Model selection & tuning 

Using the **scikit-learn Pipeline **function, we can **automate** and combine the data preprocessing, model comparison and tuning into one workflow.
