//
// Created by Robin Huang on 11/17/16.
//

#include "recycle_queue.h"
RecycleQueueDataBase::RecycleQueueDataBase()
{
    m_queue = NULL;
}

RecycleQueueDataBase::~RecycleQueueDataBase()
{

}

void RecycleQueueDataBase::recycle() const
{
    m_queue->recycle(this);
}

void RecycleQueueDataBase::push2queue() {
    m_queue->push2queue(this);
}
