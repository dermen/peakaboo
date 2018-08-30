## PEAKABOOO
```peakaboo -h```

e.g.

```peakaboo -f '/path/to/cxiDB/*.cxi' -data data ```

These are the workings of a python based peak/streak detection GUI. Py2-3 compatible.

Currently this only works for a CXIDB-style file(s) containing assembled images stored in a three-dimensional hdf5 dataset. Unassembled tools coming soon and psana navigation. 

The main peak detection algorithm is in ```peaks.py```, see the function ```pk_pos```. 

```streaks.py``` contains sub-image analysis used in streak detection and local SNR analysis for checking peak connectivity.

```imbrowse.py``` contains tools for browsing images
