
import os
import subprocess
import pdb
import swap
from optparse import OptionParser

'''
Need to run SWAP.py multiple times -- once for every "day" in GZ2
Take that output and feed it into my machine classifiers
Take that output and determine retirement

'''

parser = OptionParser()
parser.add_option("-n", "--new", action="store_true", dest="startup", 
                  default=False, help="start new SWAPSHOP with startup.config")
parser.add_option("-c","--config", dest="config_name", default=None)
#parser.add_option("-d", "--dir", dest="dirname",
#                  help="Store all output in DIR", metavar="DIR")

(options, args) = parser.parse_args()


if options.startup: config = "startup.config"
else: config = "update.config"

if options.config_name: log_dir = "logfiles_%s"%(options.config_name)
else: log_dir = "logfiles"

more_to_do = True


if not os.path.exists(log_dir): os.makedirs(log_dir)


try:
    count = int(subprocess.check_output("ls %s/ | wc -l"%log_dir, shell=True))
except: 
    count = 0

print count

while more_to_do:
    logfile = "%s/GZ2_%i.log"%(log_dir,count)

    # run SWAP.py
    os.system("python SWAP.py %s > %s"%(config,logfile))
    #os.system("python SWAP.py %s"%(config))

    # how do we get the name of the folder in which the catelogs reside?
    # take candidates from SWAP as training input for machine
    #os.system("python MachineClassifier.py -c %s"%config)

    count+=1

    #more = os.system("grep 'running' .swap.cookie | wc -l")
    more = subprocess.check_output("grep 'running' .swap.cookie | wc -l", 
                                         shell=True)

    if more=='0\n': more_to_do = False

    if more_to_do: 
        config = "update.config"
        print "Ran SWAP.py %i times"%count
        print "Still more to do!\n"
    else: print "That's the last one!"
