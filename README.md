## Notes

- MGF with spectra and SMILES
- SMILES are canonicalized using smiles-parser
- 250'000 SMILES
  


## Approach
- [ ] Binning of spectra (4096 bins)
- [ ] Check which bins are never activated
- [ ] Divide by max intensity of the spectrum
- [ ] One hot encoding on all the atoms
- [ ] Check if some atoms never appear -> the number of neuron in output layer will be equal to the number of unique atoms
- [x] use burn for DL lib, use `metal` feature
- [ ] check autotune and fusion are available for metal
- [ ] Sigmoid activation function for multiclass
- [ ] Dense layer as last layer
- [ ] Use binary cross entropy loss 
- [ ] Given that we have a highly unbalanced class, if there are atoms that always appear, they need to be "eliminated"
- [ ] If class often appears but not always, we need to add a per class weight to the loss function
- [ ] Dense/dropout (10%) /batchnorm layers -> 2048 -> 1024 -> 512 -> output layer
- [ ] Adam optimizer
- [ ] Metrics : mean MCC for all classes
- [ ] Stratified holdout 
- [ ] Use Terminal user interface of Burn