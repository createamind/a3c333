#coding: utf-8
import numpy as np
from ..utils import logger
class Memory(object):
    class Item():
        def __init__(self, state, action, reward):
            self.state = state
            self.action = action
            self.reward = reward

    def __init__(self, episode):
        self._episode = episode
        self._init()

    def _init(self):
        self._memory = []           # type: list[Memory.Item]
        self._score = 0.
        import datetime as dt
        self._timeStart = dt.datetime.now()
        self._timeEnd = None

    def add(self, state, action, reward):
        self._memory.append(self.Item(state, action, reward))
        self._score += reward

    @property
    def score(self):
        return self._score

    @property
    def episode(self):
        return self._episode

    @property
    def size(self):
        return len(self._memory)

    def __getitem__(self, item):
        return self._memory[item]

class MemorySaver(object):
    def __init__(self, save_dir, max_list = 3, min_save_score = None):
        import os
        self._save_dir = save_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        self._max_list = max_list
        self._min_save_score = min_save_score
        self._init()

    def _init(self):
        self._memoryCurrent = None
        self._memoryBests = []     # type: list[Memory]
        import os
        import filelock
        lockfile = self._save_dir + '/.lock'
        try:
            locker = filelock.FileLock(lockfile)
            with locker.acquire(30):

                for directory, dirnames, filenames in os.walk(self._save_dir):
                    for file in [os.path.join(directory, n) for n in filenames if n.endswith('.pkl')]:
                        with open(file, 'rb') as f:
                            import pickle
                            m = pickle.load(f)
                            assert(isinstance(m, Memory))
                            assert(m.episode >= 0)
                            self._memoryBests.append(m)
                            from ..utils import logger
                            logger.info("load save memory, score={:.3f}, episode={}".format(m.score, m.episode))
        except filelock.Timeout:
            from ..utils import logger
            logger.error("can not lock file {}".format(lockfile))
        pass

    def createMemory(self, episode):
        assert(self._memoryCurrent is None)
        self._memoryCurrent = Memory(episode)

    def addCurrent(self, state, action, reward, isOver):
        curMemory = self._memoryCurrent
        if curMemory:
            curMemory.add(state, action, reward)
        if not isOver: return
        if self._min_save_score is not None and curMemory.score < self._min_save_score:
            logger.info("drop memory for score {:.3f} < min_save_score {:.3f}".format(curMemory.score, self._min_save_score))
            self._memoryCurrent = None
            return
        import datetime as dt
        curMemory._timeEnd = dt.datetime.now()
        if len(self._memoryBests) >= self._max_list:
            idx_low = np.argmin([m.score for m in self._memoryBests])
            if self._memoryBests[idx_low].score < curMemory.score:
                self._removeMemory(self._memoryBests[idx_low])
                del self._memoryBests[idx_low]

        if len(self._memoryBests) < self._max_list:
            self._memoryBests.append(curMemory)
            self._saveMemory(curMemory)
            logger.info("add memory: episode={}, score={:.3f}".format(curMemory.episode, curMemory.score))

        self._memoryCurrent = None

    def _removeMemory(self, memory):
        import filelock
        lockfile = self._save_dir + '/.lock'
        try:
            locker = filelock.FileLock(lockfile)
            with locker.acquire(30):
                import os
                assert(memory.episode >= 0)
                filename = self._save_dir + '/{:04d}.pkl'.format(memory.episode)
                if os.path.exists(filename):
                    os.unlink(filename)
                logger.info("remove memory: episode={}, score={:.3f}".format(memory.episode, memory.score))
        except filelock.Timeout:
            logger.error("can not lock file {}".format(lockfile))

    def _saveMemory(self, memory):
        import filelock
        lockfile = self._save_dir + '/.lock'
        try:
            locker = filelock.FileLock(lockfile)
            with locker.acquire(30):
                import pickle
                assert(memory.episode >= 0)
                filename = self._save_dir + '/{:04d}.pkl'.format(memory.episode)
                with open(filename, 'wb') as f:
                    pickle.dump(memory, f)
        except filelock.Timeout:
            from ..utils import logger
            logger.error("can not lock file {}".format(lockfile))


