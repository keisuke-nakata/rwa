Recurrent Weighted Average model implementation & experiments

# Original paper
Machine Learning on Sequential Data Using a Recurrent Weighted Average  
Jared Ostmeyer, Lindsay Cowell  
2017  
https://arxiv.org/abs/1703.01253

# Original Keras implementation
https://gist.github.com/shamatar/55b804cf62b8ee0fa23efdb3ea5a4701  
(This repo's implemetation is almost same as the above gist script.  
I fixed the code to support `return_sequences` parameter for visualizing hidden states.)

# Note
Current implementation is heavily under development.  
Any code has not been systemically tested.  
Envitonment arguments are hard-coded.

# Usage
## settings
Please fix the `save_root_dir` variable and `if __name__ == '__main__'` section in scripts before execution!

## commands
```bash
python train.py  # train RWA (or LSTM. See and fix `__main__` section to switch the model)
python plotting.py  # plot internal states of RWA. (Before execution, check your trained model filename and fix the variables in this file!)
```

# Experimental Results
(Japanese article)
http://qiita.com/keisuke-nakata/items/f48a04d629b86bba4be5
