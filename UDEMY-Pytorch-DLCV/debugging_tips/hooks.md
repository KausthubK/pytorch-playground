
Problem: Mismatched tensor sizes e.g.:
https://stackoverflow.com/questions/49606482/how-to-resolve-runtime-error-due-to-size-mismatch-in-pytorch

I followed this tutorial on hooks to debug the problem:
https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/


```python
def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")
    for grad in i:
        try:
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad.shape)
        except AttributeError:
            print ("None found for Gradient")
    print("\n")


model.features.register_forward_hook(hook_fn)
model.classifier.register_forward_hook(hook_fn)
```

Solution was to change the tensor view by adding ```flattened_extracted_features = extracted_features.view(-1, 96)```:

```python
    def forward(self, x):
        """
        Args:
            x (PyTorch tensor): Input tensor for forward pass

        Returns: PyTorch tensor - output tensor after forward pass
        """
        extracted_features = self.features(x)

        flattened_extracted_features = extracted_features.view(-1, 96)

        classification = self.classifier(flattened_extracted_features)
        return classification
```