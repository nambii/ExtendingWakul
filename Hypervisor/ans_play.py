"""ans_play.py
To download,install and run harvester Program
Authors : Ebin Joshy Nambiaparambil,Saurabh Sharma
"""
from collections import namedtuple
import json
from ansible.executor.task_queue_manager import TaskQueueManager
from ansible.inventory import Inventory
from ansible.parsing.dataloader import DataLoader
from ansible.playbook.play import Play
from ansible.plugins.callback import CallbackBase
from ansible.vars import VariableManager
import config.core as core

from tempfile import NamedTemporaryFile
import os

class ResultsCollector(CallbackBase):

    def __init__(self, *args, **kwargs):
        super(ResultsCollector, self).__init__(*args, **kwargs)
        self.host_ok = {}
        self.host_unreachable = {}
        self.host_failed = {}

    def v2_runner_on_unreachable(self, result):
        self.host_unreachable[result._host.get_name()] = result

    def v2_runner_on_ok(self, result, *args, **kwargs):
        self.host_ok[result._host.get_name()] = result

    def v2_runner_on_failed(self, result, *args, **kwargs):
        self.host_failed[result._host.get_name()] = result
def main():
    hosts_=[]
    args = core.config('nectar','EC2')
    #set the host_list
    with open("host_created.txt","r") as f:
        lines=[line.rstrip() for line in f]
        for line in lines:
            hosts_.append(line)
    print(hosts_)
    print("Running ansible playbook")
    host_list = [hosts_[len(hosts_)-1]]
    #prepare options into tuple
    Options = namedtuple('Options', ['connection', 'module_path', 'forks', 'remote_user',
                                 'private_key_file', 'ssh_common_args', 'ssh_extra_args', 'sftp_extra_args',
                                 'scp_extra_args', 'become', 'become_method', 'become_user', 'verbosity', 'check',])
    # initialize needed objects
    variable_manager = VariableManager()
    loader = DataLoader()
    options = Options( module_path=args['module_path'],private_key_file=args['private_key_file'],connection=args['connection'], forks=100,
                  remote_user=args['remote_user'], ssh_common_args=None, ssh_extra_args=None,
                  sftp_extra_args=None, scp_extra_args=None, become=args['become'], become_method=args['become_method'],
                  become_user=args['become_user'], verbosity=None, check=False)

    passwords = dict()

    # create inventory and pass to var manager
    inventory = Inventory(loader=loader, variable_manager=variable_manager, host_list=host_list)
    variable_manager.set_inventory(inventory)

    play_source = dict(
    name="Ansible Play",
    hosts=hosts_[len(hosts_)-1],
    gather_facts='no',
    tasks = [
            dict(action=dict(module='copy', args='src=./requirements.txt dest=./ remote_src=False')),
            dict(action=dict(module='script', args='./download_packages.sh')),
            dict(action=dict(module='git', args='repo='+args['git']+' dest=./wa-harvester')),
            dict(action=dict(module='copy', args='src=./configure.yaml dest=./wa-harvester/ remote_src=False')),
            dict(action=dict(module='shell', args='python3 ' +args['harvester_py']+' ')),

         ]
    )
    print(play_source)
    play = Play().load(play_source, variable_manager=variable_manager, loader=loader)
    print(play)
    # actually run it
    tqm = None
    callback = ResultsCollector()
    try:
        tqm = TaskQueueManager(
            inventory=inventory,
            variable_manager=variable_manager,
            loader=loader,
            options=options,
            passwords=passwords,
        )
        tqm._stdout_callback = callback
        result = tqm.run(play)
        print(result)
    finally:
        if tqm is not None:
            tqm.cleanup()


    for host, result in callback.host_ok.items():
        print("UP ***********")
        print('{} >>> {}'.format(host, result.__dict__))


    for host, result in callback.host_failed.items():
        print("FAILED *******")
        print('{} >>> {}'.format(host, result.__dict__))


    for host, result in callback.host_unreachable.items():
        print("DOWN *********")
        print('{} >>> {}'.format(host, result.__dict__))
if __name__ == '__main__':
    main()
