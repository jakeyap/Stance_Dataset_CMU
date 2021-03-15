the test_set_xxx_old.bin files are tokenized using [URL]
the test_set_xxx_new.bin files are tokenized using [url]
the test_set_xxx_bertweet.bin files are tokenized using the BerTweet model's suggested tokenizer settings

the test_set_xxx_semeval17_bertweet.bin files are tokenized using the BerTweet's tokenizer. the semeval17 stance dataset


These files are a combination of the srq and semeval datasets.
merge_all_train_set_128_bertweet.bin        all the training data
merge_all_test_set_128_bertweet.bin         all the test data combined
merge_semeval_test_set_128_bertweet.bin     only semeval's test data
merge_srq_test_set_srq_128_bertweet.bin     only srq's test data

===========================================
train_set_128_w_length_bertweet.bin (length 2766)
test_set_128_w_length_bertweet.bin (length 692)
These files are the stance dataset, but filtered. Rows with missing txt are removed.
Rows which have their meta data missing on twitter are also removed.

For the TRAINING SET, if you want to split virality based on LIKES, the thresholds are as follows
10% : 4         70% : 1415
20% : 22        80% : 3556
30% : 54        90% : 11236
50% : 291.5

For the TRAINING SET, if you want to split virality based on RETWEETS, the thresholds are as follows
10% : 1         70% : 587
20% : 8         80% : 1417
30% : 24        90% : 4260
50% : 128.5
===========================================
train_set_128_individual_bertweet.bin (length 2761)
test_set_128_individual_bertweet.bin (length 690)

When this data was hydrated, there were additional tweets that were removed. 

These files are the stance dataset, but with some major changes from previous files. 
1. Tweets are encoded separately instead of pairwise
2. Rows with missing txt are removed.
3. Includes extra data about the users (username and followers_count)

Rows which have their meta data missing about the target post on twitter are also removed.
For rows with missing response tweet meta data, they are filled in with "unknown user" for username and 0 for follower count. If the root tweet's meta data is missing, the post is dropped.
