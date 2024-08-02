#!/bin/bash

# test on first 8k
#python3 main.py --isnewsave --outputmode 8 --gpu 1 --purpose jonasTest_with10kSamplesOnlyAtlasSuperLong --batch_size 32 --num_workers 5 --starttrain 0 --endtrain 8000 --startval 8000 --endval 9984 --dropoutrate 0.0 --lr 0.00006 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.05 --savelogdir="./result/" --num_epochs 150

# test on 25k
#python3 main.py --isnewsave --outputmode 8 --gpu 3 --purpose jonasTest_with25kSamplesDiffBGTissueAll --batch_size 32 --num_workers 5 --starttrain 0 --endtrain 28000 --startval 28000 --endval 29984 --dropoutrate 0.0 --lr 0.00006 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.05 --savelogdir="./result/" --num_epochs 100

# test on 3k
#python3 main.py --isnewsave --outputmode 8 --gpu 3 --purpose jonasTest_with3kSamplesDiffBGTissueAll --batch_size 32 --num_workers 5 --starttrain 0 --endtrain 3000 --startval 28000 --endval 29984 --dropoutrate 0.0 --lr 0.00006 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.05 --savelogdir="./result/" --num_epochs 70

# test on 25k finetune
#python3 main.py --outputmode 8 --gpu 3 --purpose jonasTest_finetune_with25kSamplesDiffBGTissueAll --batch_size 32 --num_workers 5 --starttrain 0 --endtrain 28000 --startval 28000 --endval 29984 --dropoutrate 0.0 --lr 0.00006 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.05 --savelogdir="./result/" --num_epochs 10 --finetune --loaddir "/home/home/jonas/programs/learn-morph-infer/log/1403-15-48-19-v7_1-jonasTest_with25kSamplesDiffBGTissueAll/epoch49.pt" --thFlair 0.1 --thT1 0.6

# test on 3k finetune 
python3 main.py --outputmode 8 --gpu 1 --purpose jonasTest_finetune_with25kSamplesDiffBGTissueAll --batch_size 32 --num_workers 5 --starttrain 0 --endtrain 3000 --startval 28000 --endval 29984 --dropoutrate 0.0 --lr 0.00006 --lr_scheduler_rate 0.999997 --weight_decay_sgd 0.05 --savelogdir="./result/" --num_epochs 10 --finetune --loaddir "/home/home/jonas/programs/learn-morph-infer/log/1403-15-48-19-v7_1-jonasTest_with25kSamplesDiffBGTissueAll/epoch49.pt" --thFlair 0.1 --thT1 0.6
