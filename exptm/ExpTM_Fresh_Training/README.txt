Extract this complete ExpTM_Fresh_Training folder inside your existing exptm folder.
Open 02_Train_ExpTM_From_Scratch.ipynb and select the existing ExpTM environment.
Restart the kernel before beginning. Run Cells 1-3 first, then Cell 4 to train.
Cell 5 performs the small sampling pilot. Do not run the old Colab or overwrite old runs.
Default: 5 epochs, 10,000 supplied configurations, batch 64, fresh 3-channel model.
Each completed epoch saves a checkpoint in runs/<unique_run>/models/.
No torch runtime testing has been performed in the delivery workspace; see notebook validation note.
