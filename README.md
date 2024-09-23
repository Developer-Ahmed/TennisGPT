# Train your own GPT
1. Modify the config file to your needs. Make sure to set pretraining to True when training and false when finetuning.
2. Run training.py
3. Change config file for finetuning. Set pretraining to false.
4. Run chat.py to talk to the finetuned model.
# Train your own TennisGPT
Same steps above except when finetuning you can use a custom csv file with columns Question and Answer (case-sensitive).
You don't need to include special tokens in the csv file, load_data handles that.
# How to set the config file?
You can take a look at the example of pretraining above named config_example.py
   
