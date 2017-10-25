//
// Created by Robin Huang on 03/17/17.
//

#ifndef APPC_BATCH_DATA_H
#define APPC_BATCH_DATA_H

#include "stdafx.h"
#include <queue>

class RecycleQueueBase;

class RecycleQueueDataBase
{
public:
    RecycleQueueDataBase();
    virtual ~RecycleQueueDataBase();
    virtual void recycle() const;
    virtual void push2queue();
    RecycleQueueBase * queue() const { return m_queue; };
protected:
    friend class RecycleQueueBase;
    RecycleQueueBase * m_queue;
    int m_batch_size;
};

class RecycleQueueBase
{
public:
protected:
    friend class RecycleQueueDataBase;
    virtual void recycle(const RecycleQueueDataBase * p) = 0;
    virtual void push2queue(RecycleQueueDataBase * p) = 0;
    void setDataQueue(RecycleQueueDataBase * data) {
        data->m_queue = this;
    }
};

template <typename T>
class RecycleQueue : public RecycleQueueBase
{
public:
    RecycleQueue(uint32_t maxsize, const QString name = "")
    : m_name(name), m_maxsize(maxsize)
    {
        flagEnd = false;
        for(uint32_t i = 0; i < maxsize; i++)
        {
            T * data = new T();
            setDataQueue(data);
            m_emptyQueue.push_back(data);
            m_queue_all.push_back(std::shared_ptr<T>(data));
        }
    }
    ~RecycleQueue()
    {
        m_queue_all.clear();
    }

    int sizeMax() const { return m_queue_all.size(); };

    int size() const { return m_queue.size();};

    int sizeEmpty() const { return m_emptyQueue.size(); };

    bool isFull() const { return m_queue.size() == m_maxsize; };

    bool hasNoEmpty() const { return m_emptyQueue.size() == 0; };

    T * pop(int timeout = -1)
    {
        while (!flagEnd)
        {
            while (timeout < 0 && !flagEnd && m_queue.size() == 0) {
                m_semDataAvaiable.acquire();
            }
            if(flagEnd) break;
            QMutexLocker locker(&m_mutexQueue);
            if (m_queue.size() > 0)
            {
//            MSP_INFO("pop data, queue size = %d, emptyQueue size = %d", m_queue.size(), m_emptyQueue.size());
                T * data =  m_queue.front();
                m_queue.pop_front();
                return data;
            }
            if(timeout >= 0) break;
        }
        return NULL;
    }

    void push(T* data)
    {
        Q_ASSERT(data);
        while (!flagEnd && m_queue.size() >= m_maxsize) {
//            MSP_INFO("queue %s push wait, size=%d[%d/%d]", m_name.toStdString().c_str(), m_queue.size(), m_emptyQueue.size(), m_maxsize);
            m_semEmptyDataAvaiable.acquire();
        }

        if(flagEnd) return;

        m_mutexQueue.lock();
        m_queue.push_back(data);
        m_mutexQueue.unlock();
//        MSP_INFO("push data, queue size = %d, emptyQueue size =%d", m_queue.size(), m_emptyQueue.size());
        m_semDataAvaiable.release();
    }

    virtual void push2queue(RecycleQueueDataBase * p) {
        push((T*)p);
    }

    T * getEmpty()
    {
        uint32_t retry = 0;
        T * ret = NULL;
        while(!flagEnd) {
            while (!flagEnd && m_emptyQueue.size() == 0) {
                retry++;
                m_semEmptyDataAvaiable.acquire();
            }
            if (flagEnd) return NULL;
            if (retry > 0)
            {
//                MSP_WARNING("queue %s retry for getEmpty, retry=%d, size=%d[%d/%d]", m_name.toStdString().c_str(), retry, m_queue.size(), m_emptyQueue.size(), m_maxsize);
            }
            QMutexLocker locker(&m_mutexEmptyQueue);
            if(m_emptyQueue.size() > 0) {
                T *ret = m_emptyQueue.front();
                m_emptyQueue.pop_front();
                return ret;
            }
        }
        return ret;
    }

    void close()
    {
        flagEnd = true;
        m_semDataAvaiable.release();
        m_semEmptyDataAvaiable.release();
    }
protected:
    friend class RecycleQueueDataBase;
    QString m_name;
    uint32_t m_maxsize;
    QQueue<T*> m_queue;
    QMutex m_mutexEmptyQueue;
    QQueue<T*> m_emptyQueue;
    std::vector<std::shared_ptr<T>> m_queue_all;
    QMutex m_mutexQueue;
    QSemaphore m_semDataAvaiable, m_semEmptyDataAvaiable;
    bool flagEnd;

    virtual void recycle(const RecycleQueueDataBase * bd)
    {
        Q_ASSERT(bd->queue() == this);
        m_mutexEmptyQueue.lock();
        m_emptyQueue.push_back((T*)bd);
//    MSP_INFO("recycle, emptyQueue size = %d, queue size = %d", m_queue->m_emptyQueue.size(), m_queue->m_queue.size());
        m_mutexEmptyQueue.unlock();
        m_semEmptyDataAvaiable.release();
    }
};

#endif //APPC_BATCH_DATA_H
