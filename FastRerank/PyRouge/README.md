# PyRouge
Rouge evaluation script implemented with Python.

Currently, only Rouge-N is implemented.

**WARNING** The result is slightly different from the official Rouge.

## Usage

```python
python compute.py ref_file.txt predict_file.txt
```

`ref_file.txt` and `predict_file.txt` are line-by-line text files.

Output format

Now the script returns a dictionary, which looks like:
```python
{'rouge-1': {'p': (0.34902417721047307, 0.0013577881868447896, (0.34636268087782229, 0.35168567354312386)), 'r': (0.29738279969648435, 0.0011050260347502225, (0.29521676027482341, 0.29954883911814528)), 'f': (0.31108022747945868, 0.0010620266366127937, (0.30899847420902349, 0.31316198074989388))}, 'rouge-2': {'p': (0.13283309482481312, 0.0010693069735949634, (0.13073707085268366, 0.13492911879694258)), 'r': (0.11229619796675784, 0.00089595545126339876, (0.11053997253273248, 0.1140524234007832)), 'f': (0.11772894246560359, 0.00090995790892731512, (0.11594526982730757, 0.1195126151038996))}}
```
For each measurement, the result is a tuple, which is (mead, std_error, (95% confidence interval))