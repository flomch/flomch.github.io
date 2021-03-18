<!-- Output copied to clipboard! -->

<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 0.916 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Wed Mar 17 2021 17:25:57 GMT-0700 (PDT)
* Source doc: Oh gosh
* Tables are currently converted to HTML tables.
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 1.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



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
