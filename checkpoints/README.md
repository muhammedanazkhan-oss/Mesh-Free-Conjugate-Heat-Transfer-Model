# Trained model checkpoints

Training to the accuracy reported in the paper uses a GPU in double precision
(FP64). Run:

    python src/pinn_mchs.py

and extend the script to save `model.state_dict()` into this folder as
`pinn_mchs_baseline.pt`. Deposit the trained weights here before archiving the
release on Zenodo so that the results are fully reproducible without retraining.
