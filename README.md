# Credit Underwriting Technical Presentation

For more personal insights, please refer to the threads by Olivier, Ryan, and myself, as well as Phil's comment.

## Introduction

From my prior experience with credit underwriting, I’ve come to appreciate that this is one of the most complex problems for applying machine learning. Data in this domain tends to be very heterogeneous, collected over different time frames, and coming from many different sources that may change and alter in the midst of the data collection process. Coming up with a proper target variable is also a very tricky process, requiring deep domain knowledge and refined business analysis skills. I want to commend Home Credit and Kaggle for providing such a great dataset, which was leak-free and very amenable to machine learning techniques.

Based on what is known about credit underwriting, and similar machine learning problems, it was clear that two things would be crucial for building a good model for this competition:
1. A good set of smart features.
2. A diverse set of base algorithms.

We were able to utilize four main sources of feature diversity, along with a few minor additional ones.
