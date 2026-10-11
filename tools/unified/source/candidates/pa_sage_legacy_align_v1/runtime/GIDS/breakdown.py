sample_count = 0
# ns
sample_time = 0
fetch_feature_count = 0
# ns
fetch_feature_time = 0

def sample_record(_sample_time):
    global sample_count
    global sample_time
    sample_time += _sample_time
    sample_count += 1

def fetch_feature_record(_fetch_feature_time):
    global fetch_feature_count
    global fetch_feature_time
    fetch_feature_time += _fetch_feature_time
    fetch_feature_count += 1

def get_sample_record():
    #return sample_count,fetch_feature_count
    return sample_count, sample_time

def get_fetch_feature_record():
    return fetch_feature_count,fetch_feature_time