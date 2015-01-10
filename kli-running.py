#running:
#Called from kli-supervisor
#Pass arg: user@host
#Logs into remote computer via ssh
#If no args passed performs operation on localhost
#Does a ps -agx
#Pulls off and counts kli-server kli-worker
#Logs out, if logged in
#Returns tuple (servers, workers)

#launchServer:
#called from kli-supervisor
#Pass arg: user@host
#No arg: localhost

#launchWorkers:

#processesRemaining():
#returns number of processors - number of workers (or 0 if this quantity is negative)

#killWorkers

#killServer

#  allRunning:
#    called from kli-supervisor
#    same as running except uses list of available hosts:
#  launchAll:
#  killAll:

###########Functional test##############:
#On localhost; assert running(no args) == (0,0)  #Must not have any kli-processes running
#launchServer(no args)
#assert running(no args) == (1,0)
#launchWorkers(no args)  # launchers 1 worker
#assert running(no args) == (1,1)
#launchWorkers(numWorkers=processesRemaining())
#assert running(no args) = (1, numproc)
#print out numproc
#killWorkers()
#assert running() = (1,0)
#killServer()
#assert running() = (0,0)

#try with: sean@127.0.0.1
#try with: host list

while True:
    pass