import supervisor

def printRunning(R):
    print 'Running:', R[0], 'Servers and', R[1], 'Workers'
    
print 'Verify nothing is running'    
R = supervisor.running()
assert(R[0]==0)
assert(R[1]==0)
printRunning(R)

print 'Launching server'
supervisor.launchServer()

print 'Verify server and nothing else is running'
R = supervisor.running()
assert(R[0]==1)
assert(R[1]==0)
printRunning(R)

###########Functional test##############:
#On localhost;
#assert running(no args) == (0,0)  #Must not have any kli-processes running
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
