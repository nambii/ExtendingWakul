"""create_vm.py
Program to create vm and run ans_play.py
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




reservation = ec2_conn.run_instances(args2['image_name'], key_name=args2['key_name'],instance_type=args2['instance_type'], security_groups=args2['security_groups'],placement=args2['placement'])


instance = reservation.instances[0]


print ('waiting for instance_1')
while instance.state != 'running':
    print ('.')
    time.sleep(5)
    instance.update()

print ('done')
time.sleep(10)
print(instance.__dict__)
#saving ip_address
f = open('host_created.txt', 'a')
file_contents = f.write((instance.__dict__['private_ip_address']))
file_contents = f.write('\n')
hosts_=[]
with open("host_created.txt","r") as f:
    lines=[line.rstrip() for line in f]
    for line in lines:
        hosts_.append(line)
#saving instance_d
f = open('instances.txt', 'a')
file_contents = f.write((instance.__dict__['id']))
file_contents = f.write('\n')
print('ip-address :' ,hosts_[len(hosts_)-1])
print('instance_id:' ,instance.__dict__['id'])
time.sleep(60)
print("Running ansible playbook")
host_list = [hosts_[len(hosts_)-1]]
#adding to known host list
os.system('ssh-keyscan '+ hosts_[len(hosts_)-1] +' >> ~/.ssh/known_hosts')
os.system('python3 ans_play.py')
