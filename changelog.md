* Changed epochs (2 -> 1500) as per the paper
* Disabled TF logs
* Created a mode to loop through whole UCR
* Created a mode to disable validation monitoring to simulate the paper evaluation  
\# Baseline benchmark 
* Refactored the codecarbon implementation for performance
* Improved the TF implementation using low level settings (eg mixed precision)
* Created the dataset subset:
The best approach is to stratify by the key axes that affect compute time and model behavior: dataset size (number of training samples), time series length, and number of classes. Here's a curated subset of ~30 datasets that covers the full spectrum:

Adding Fast Benchmark Dataset Subset
Retrieved metadata for all 128 UCR datasets. Training sizes range from 16 to 8926, lengths from 15 to 2709, classes from 2 to 60.

Here's the best approach for a representative fast subset. I'd pick ~30 datasets stratified across 3 axes:

Category	Train Size	Example Picks
Tiny (≤40)	16–40	Coffee, Beef, GunPoint, ECG200, Wine, ToeSegmentation1
Small (41–150)	41–150	ArrowHead, CBF, BME, FreezerRegularTrain, Trace, Plane
Medium (151–500)	151–500	FacesUCR, Fish, Yoga, SwedishLeaf, ECG5000, ChlorineConcentration, Adiac, FiftyWords
Large (500+)	500+	FaceAll, StarLightCurves, Wafer, FordA, ElectricDevices, Crop, NonInvasiveFetalECGThorax1
This gives a mix of:

Short TS (15–150): SmoothSubspace, GunPoint, ECG200, SwedishLeaf
Medium TS (150–500): Fish, Yoga, ArrowHead, FordA
Long TS (500+): StarLightCurves, Trace, HandOutlines
Few classes (2–3): Wafer, Coffee, GunPoint
Many classes (10+): FacesUCR, Adiac, FiftyWords, Crop

* Optimized saving using save_weights_only vs full model (reduces time ~5%)

* Tested:
- AdamW -> Shows improvement
- BatchSize: (32 and 128 vs 64 = baseline) -> lower batch size = better accuracy but significantly longer training time
- global znormalization: 1% better accuracy (vs per sample)
- global min-max: 1% worse accuracy (vs og = per sample)
- added differenced input channel: 1.4% better accuracy (params below)

AFTER ADDING DIFFERENCED CHANNEL
==================================================
Classifier: LITE
Dataset:    Coffee
Dimensions: 286 (length) x 2 (channels)
Classes:    2
--------------------------------------------------
Total Parameters:         13,350
Trainable Parameters:     12,120
Non-trainable Parameters: 1,230
==================================================


BEFORE:
==================================================
Classifier: LITE
Dataset:    Coffee
Dimensions: 286 (length) x 1 (channels)
Classes:    2
--------------------------------------------------
Total Parameters:         10,672
Trainable Parameters:     9,880
Non-trainable Parameters: 792
==================================================

- Added indepedent learning to custom kernels vs fronzen (800 more learnable params) +1.4% acc.

==================================================
Classifier: LITE
Dataset:    Coffee
Dimensions: 286 (length) x 2 (channels)
Classes:    2
--------------------------------------------------
Total Parameters:         13,350
Trainable Parameters:     12,996
Non-trainable Parameters: 354
==================================================