# instant_tsc


This is a supporting site for the source code of the paper 'Instant Time-Series Classification via Deep Multi-Task Learning' by Josif Grabocka and Lars Schmidt-Thieme.


In order to run the code:

1 - Clone the repository locally


2 - We provided the smallest dataset HAR for testing the code, the other datasets exceed the Github file upload limit. Unzip the *.zip files in the folders 'HAR/0/', 'HAR/1/' and 'HAR/2/'.

3 - To run the code for the first split of the HAR dataset for the truncated version with a demanded early franction of 0.1, issue:

python3 HAR/0/ singletask 0.1 

for the multitask versions with decay or gaussian smoothing, issue

python3 HAR/0/ multitask 0.1 decay

python3 HAR/0/ singletask 0.1 norm

