In addition to basic baselines that would be important when trying to accomplish various tasks with a sparse autoencoder (peft, model editing, rep-e, probing, data screening), I think that a clustering method could offer a pretty compelling baseline.

Approach:
1. Take the same layer the SAE was trained on and get the neural activations in that layer across your SAE training dataset.
2. Normalize/whiten the activations.
3. Perform some clustering alg (K-means, HAC, a probabilistic alg, etc) on the points with the same number, n, of clusters as dimensions in the SAE.
4. Represent each cluster using some kernel density model or something like that.
5. Represent each example as an n-dimensional vector of its densities (or some other cluster membership measure).

I think that anything you might want to do with an SAE, you can do with clusters:
- Characterizing "features": each cluster could be considered a feature and could be labeled by humans or a chatbot based on exemplars (in exactly the same way SAE neurons are labeled).
- Editing: a method for editing the model's activations would be to move an activation in the gradient of a target cluster's KDM.
- Anomaly detection: an example that has a low-norm vector representation can be flagged as an anomaly.
- PEFT: There are many possibilities. One would be to parameterize training with n weights which each specify how much to move activations in the direction of each cluster (according to some metric or step-taking rule).
- LAT: you can parameterize attacks in the same way you can parameterize PEFT.

Advantages:
- This would be much simpler than SAEs.
- This could be much faster to fit (but some batched clustering approach might be needed).
- It would probably be better for anomaly detection.
- This would be sensitive to instances in which two different concepts were represented with activations of the same direction but different magnitudes.