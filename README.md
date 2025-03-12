I have organized the scripts such that all the scripts are in one folder, and all the .csv files are in another folder. 

** MAKE SURE YOUR .CSV FILES ARE IN THE FOLDER, OTHERWISE THE SCRIPTS WON'T RUN PROPERLY!!! **

I have added a script that adds all the original columns to the processed data. So, once you have processed the data through the processtext.py script, then run the addcolumn.py script.
As well, I blieve the wordfrequency.py script is implemented correctly, and assuming so, it will add a column to the .csv file with an array for each article giving the frequency of each most used word. This should be able to be used in the logistical regression model to speed up the process.
