import sagemaker
import boto3
import pandas as pd


sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'awsthesishsenvbucket'
print('using bucket:' + bucket)



# df = pd.read_csv('/Users/ramtahor/Desktop/awscsv.csv')


sk_prefix = 'sagemaker/hyper_spectral_data'
try_path = sess.upload_data(
    path = '/Users/ramtahor/Volumes/Extreme Pro/pair_new', bucket=bucket, key_prefix=sk_prefix
)
