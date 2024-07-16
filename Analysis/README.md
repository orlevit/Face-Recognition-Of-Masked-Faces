## Summary
Brief summary of the analysis results.


🧪 Testing scenarios:
<br><br>
Two scenarios of Face verification were tested on the Sunglasses, Hat, Surgical and Ski masks:

a.  Masked-Masked: A comparison between two masked facial images.

b.  Masked-No Masked: A comparison between a masked facial image and an unmasked facial image.

<p align="center">
  <img src="images/5.jpg" width="550" height="110">
</p>

📊Analysis Findings:

We conducted an extensive analysis on the LFW and AgeDB30 benchmarks and the synthetically occluded version of them. Since we had a large amount of data, as shown in the ROC plots for the masked-masked scenario below.
<p align="center">
  <img src="images/6.jpg" width="1000" height="500">
</p>

We introduced a metric to obtain a single value for each model, aiming to enhance our understanding of the results. We named the metric - “Mean Average AUC”, calculated as follows:
<p align="center">
  <img src="images/7.jpg" width="350" height="300">
</p>

Here are the results of our study:
1.	When the type of occlusion is not known in advance:
<p align="center">
  <img src="images/8.jpg" width="380" height="140">
</p>

2.	When the type of occlusion is known in advance:
<p align="center">
  <img src="images/9.jpg" width="350" height="80">
</p>

🎯Summary and Conclusions:

1.	Notably, fine-tuning the model based on a known occlusion type led to significant improvements.

2.	Interestingly, even out-of-the-box face recognition systems achieved high accuracy in identifying occluded faces but fine-tuning on occluded face images further enhanced performance.
