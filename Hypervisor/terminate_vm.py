"""terminate_vm.py
To terminate a VM in fcfs mode
Authors : Ebin Joshy Nambiaparambil,Saurabh Sharma
"""
import boto
import time
from boto.ec2.regioninfo import RegionInfo
import config.core as core
import os
import subprocess
args1 = core.config('nectar','EC2')
args2 = core.config('nectar','VM')
region=RegionInfo(name=args1['name'], endpoint=args1['endpoint'])
ec2_conn = boto.connect_ec2(aws_access_key_id=args1['aws_access_key_id'],
aws_secret_access_key=args1['aws_secret_access_key'], is_secure=args1['is_secure'],
region=region, port=args1['port'], path=args1['path'], validate_certs=args1['validate_certs'])

instances_=[]
with open("instances.txt","r") as f:
    lines=[line.rstrip() for line in f]
    for line in lines:
        instances_.append(line)

print(instances_)
instance=instances_[0]
print(instance)
reservation = ec2_conn.terminate_instances(instance_ids=[instance])
