from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import time
import os
import numpy as np
from Dataset import Dataset
from model import DDEA
import time
from utils import *
from scipy.sparse import csr_matrix

name = 'ddea'
topK_list = [5, 10]

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=256, help='batch size.')
parser.add_argument('--emb_size', type=int, default=200, help='embed size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--log', type=str, default='logs/{}'.format(name), help='log directory')
parser.add_argument('--self', type=float, default=1.0, help='single domain loss ratio')
parser.add_argument('--cross', type=float, default=1.0, help='cross domain loss ratio')
parser.add_argument('--opk', type=float, default=0.8, help='overlapping ratio')
parser.add_argument('--lam',type=float,default=0.1, help='local embedding alignment loss ratio')
parser.add_argument('--beta',type=float,default=0.1, help='global embedding alignment loss ratio')
parser.add_argument('--dataset', type=str, default='amazon', help='amazon')
parser.add_argument('--target_percent', type=float, default=1.0, help='target percent')
parser.add_argument('--source_percent', type=float, default=1.0, help='source percent')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()


def main():
    log = os.path.join(args.log, '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.emb_size, args.weight_decay, args.self,args.opk, args.lam, args.beta, ))

    os.makedirs(log)

    print('Downloading the dataset...')
    dataset = Dataset(args.batch, dataset=args.dataset)

    NUM_USER = dataset.num_user
    NUM_MOVIE = dataset.num_movie
    NUM_BOOK = dataset.num_book

    print('Preparing the training data...')
    # Training data for domain a
    row, col = dataset.get_part_train_indices('movie', args.source_percent)
    values = np.ones(row.shape[0])
    user_x = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray()
    # Training data for domain b
    row, col = dataset.get_part_train_indices('book', args.target_percent)
    values = np.ones(row.shape[0])
    user_y = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()

    # User one-hot representations for all users (including users in domain a and domain b)
    user_id = np.arange(NUM_USER).reshape([NUM_USER, 1])
    # Split the user set
    per = (1.0-args.opk)/2.0
    user_id_1 = np.arange(NUM_USER- int(per*NUM_USER)).reshape([NUM_USER- int(per*NUM_USER),1])
    user_id_2 = np.arange(int(per*NUM_USER),NUM_USER).reshape([NUM_USER- int(per*NUM_USER),1])
    user_id_3 = np.arange(int((1-args.opk)/2.0*NUM_USER)).reshape([int((1-args.opk)/2.0*NUM_USER),1])
    user_id_4 = np.arange(NUM_USER-int((1-args.opk)/2.0*NUM_USER),NUM_USER).reshape([int((1-args.opk)/2.0*NUM_USER),1])
    # Classify the users : common or not
    user_domain1 = set(np.arange(int((1-args.opk)/2.0*NUM_USER)).tolist())
    user_domain2 = set(np.arange(NUM_USER-int((1-args.opk)/2.0*NUM_USER),NUM_USER).tolist())
    user_common = set(np.arange(int((1-args.opk)/2.0*NUM_USER),NUM_USER-int((1-args.opk)/2.0*NUM_USER)).tolist())

    user_x = torch.FloatTensor(user_x)
    user_y = torch.FloatTensor(user_y)

    # Prepare the train loaders
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),
                                                     batch_size=args.batch,
                                                     shuffle=True)
    train_loader_1 = torch.utils.data.DataLoader(torch.from_numpy(user_id_1),
                                                     batch_size=args.batch,
                                                     shuffle=True)
    train_loader_2 = torch.utils.data.DataLoader(torch.from_numpy(user_id_2),
                                                     batch_size=args.batch,
                                                     shuffle=True)
    train_loader_3 = torch.utils.data.DataLoader(torch.from_numpy(user_id_3),
                                                     batch_size=args.batch,
                                                     shuffle=True)
    train_loader_4 = torch.utils.data.DataLoader(torch.from_numpy(user_id_4),
                                                     batch_size=args.batch,
                                                     shuffle=True)
        
    pos_weight = torch.FloatTensor([args.pos_weight])
    if args.cuda:
        pos_weight = pos_weight.cuda()

    model = DDEA(NUM_USER=NUM_USER, NUM_MOVIE=NUM_MOVIE, NUM_BOOK=NUM_BOOK,
                 EMBED_SIZE=args.emb_size, dropout=args.dropout)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    BCEWL = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    LL2 = torch.nn.MSELoss(reduction='sum')
    
    if args.cuda:
        model = model.cuda()
    
    # Prepare for testing process
    movie_vali, movie_test, movie_nega = dataset.movie_vali, dataset.movie_test, dataset.movie_nega
    book_vali, book_test, book_nega = dataset.book_vali, dataset.book_test, dataset.book_nega
    feed_data = {}
    feed_data['fts1'] = user_x
    feed_data['fts2'] = user_y
    feed_data['movie_vali'] = movie_vali
    feed_data['book_vali'] = book_vali
    feed_data['movie_test'] = movie_test
    feed_data['book_test'] = book_test
    feed_data['movie_nega'] = movie_nega
    feed_data['book_nega'] = book_nega

    best_hr1, best_ndcg1, best_mrr1 = 0.0, 0.0, 0.0
    best_hr2, best_ndcg2, best_mrr2 = 0.0, 0.0, 0.0
    val_hr1_list, val_ndcg1_list, val_mrr1_list = [], [], []
    val_hr2_list, val_ndcg2_list, val_mrr2_list = [], [], []

    loss_list = []
    local_loss_list = []
    global_loss_list = []
    epoch_time_list = []


    for epoch in range(args.epochs):
        model.train()
        
        batch_loss_list = []
        batch_local_loss_list = []
        batch_global_loss_list = []
        
        epoch_time = 0.0
        for batch_idx, (data1,data2) in enumerate(zip(train_loader_1,train_loader_2)):
            
            data1 = data1.reshape([-1])
            data2 = data2.reshape([-1])
            list_1 = data1.numpy().tolist()
            list_2 = data2.numpy().tolist()
            common_set = set(list_1) & set(list_2)

            in_com_1 = []
            in_com_2 = []
            out_com_1 = []
            out_com_2 = []
            for idx,user in enumerate(list_1):
                if user in common_set:
                    in_com_1.append(idx)
                else:
                    out_com_1.append(idx)
            for idx,user in enumerate(list_2):
                if user in common_set:
                    in_com_2.append(idx)
                else:
                    out_com_2.append(idx)
            
            in_com_1 = torch.from_numpy(np.array(in_com_1))
            in_com_2 = torch.from_numpy(np.array(in_com_2))
            out_com_1 = torch.from_numpy(np.array(out_com_1))
            out_com_2 = torch.from_numpy(np.array(out_com_2))
            
            
            optimizer.zero_grad()

            if args.cuda:
                batch_user_1 = data1.cuda()
                batch_user_2 = data2.cuda()
                batch_user_x = user_x[data1].cuda()
                batch_user_x2y = user_y[data1].cuda()
                batch_user_y = user_y[data2].cuda()
                batch_user_y2x = user_x[data2].cuda()
                in_com_1 = in_com_1.cuda()
                in_com_2 = in_com_2.cuda()
                out_com_1 = out_com_1.cuda()
                out_com_2 = out_com_2.cuda()

            else:
                batch_user_1 = data1
                batch_user_2 = data2
                batch_user_x = user_x[data1]
                batch_user_x2y = user_y[data1]
                batch_user_y = user_y[data2]
                batch_user_y2x = user_x[data2]


            time1 = time.time()
            pred_x, pred_y, pred_x2y, pred_y2x, z_x, z_y, z_x_dual_loss, z_y_dual_loss = model.forward(batch_user_1,batch_user_2, batch_user_x, batch_user_y)
            time2 = time.time()
            epoch_time += time2 - time1
            loss_x = BCEWL(pred_x, batch_user_x).sum()
            loss_y = BCEWL(pred_y, batch_user_y).sum()
            # loss_x2y = 0
            # loss_y2x = 0
            # if len(common_set) >0:
            #     loss_x2y = BCEWL(pred_x2y[in_com_1], batch_user_x2y[in_com_1]).sum()
            #     loss_y2x = BCEWL(pred_y2x[in_com_2], batch_user_y2x[in_com_2]).sum()
            loss_local = 0

            if len(common_set) >0:
              loss_local = LL2(z_x[in_com_1],z_y[in_com_2])
            
            loss_global = mmd_loss(torch.mean(z_x,axis=0),torch.mean(z_y,axis=0)) + coral_loss(z_x,z_y)
            
            
            #loss = args.self*(loss_x + loss_y) + args.lam * (para_k * loss_local + (1 - para_k) * loss_global) + args.cross*(loss_x2y+loss_y2x) 
            loss = args.self*(loss_x + loss_y) + args.lam * loss_local + args.beta * loss_global
            loss.backward()
            optimizer.step()
            

            batch_loss_list.append(loss.item())
            batch_local_loss_list.append(loss_local.item())
            batch_global_loss_list.append(loss_global.item())

        epoch_time_list.append(epoch_time)
        epoch_loss = np.mean(batch_loss_list)
        epoch_local_loss = np.mean(batch_local_loss_list)
        epoch_global_loss = np.mean(batch_global_loss_list)
        loss_list.append(epoch_loss)
        local_loss_list.append(epoch_local_loss)
        global_loss_list.append(global_loss_list)
        print('epoch:{}, loss:{:.4f}, local loss:{:.4f}, global loss:{:.4f}'.format(epoch,epoch_loss,epoch_local_loss,epoch_global_loss))

        if epoch % 1 == 0:
            model.eval()

            avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2 = test_process(model, train_loader_3,train_loader_4, feed_data,
                                                                                      args.cuda, topK_list[1], mode='val')

            val_hr1_list.append(avg_hr1)
            val_ndcg1_list.append(avg_ndcg1)
            val_mrr1_list.append(avg_mrr1)
            val_hr2_list.append(avg_hr2)
            val_ndcg2_list.append(avg_ndcg2)
            val_mrr2_list.append(avg_mrr2)

            print('test: movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
                  .format(avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2))
            with open(log + '/tmp.txt', 'a') as f:
                f.write('test: movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
                        .format(avg_hr1, avg_ndcg1, avg_mrr1, avg_hr2, avg_ndcg2, avg_mrr2))

            if avg_hr1 > best_hr1:
                best_hr1 = avg_hr1
                torch.save(model.state_dict(), os.path.join(log, 'best_hr1.pkl'))

            if avg_ndcg1 > best_ndcg1:
                torch.save(model.state_dict(), os.path.join(log, 'best_ndcg1.pkl'))
                best_ndcg1 = avg_ndcg1
            if avg_mrr1 > best_mrr1:
                torch.save(model.state_dict(), os.path.join(log, 'best_mrr1.pkl'))
                best_mrr1 = avg_mrr1
            if avg_hr2 > best_hr2:
                torch.save(model.state_dict(), os.path.join(log, 'best_hr2.pkl'))
                best_hr2 = avg_hr2
            if avg_ndcg2 > best_ndcg2:
                torch.save(model.state_dict(), os.path.join(log, 'best_ndcg2.pkl'))
                best_ndcg2 = avg_ndcg2
            if avg_mrr2 > best_mrr2:
                torch.save(model.state_dict(), os.path.join(log, 'best_mrr2.pkl'))
                best_mrr2 = avg_mrr2


    print('best val movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
          .format(best_hr1, best_ndcg1, best_mrr1, best_hr2, best_ndcg2, best_mrr2))
    with open(log + '/tmp.txt', 'a') as f:
        f.write('best val movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
          .format(best_hr1, best_ndcg1, best_mrr1, best_hr2, best_ndcg2, best_mrr2))

    
    print('Val process over!')
    print('Test process......')
    for topK in topK_list:
        model.load_state_dict(torch.load(os.path.join(log, 'best_hr1.pkl')))
        test_hr1, _, _, _, _, _ = test_process(model, train_loader_3,train_loader_4, feed_data, args.cuda, topK, mode='test')

       
        model.load_state_dict(torch.load(os.path.join(log, 'best_ndcg1.pkl')))
        _, test_ndcg1, _, _, _, _ = test_process(model, train_loader_3,train_loader_4, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_mrr1.pkl')))
        _, _, test_mrr1, _, _, _ = test_process(model, train_loader_3,train_loader_4, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_hr2.pkl')))
        _, _, _, test_hr2, _, _ = test_process(model, train_loader_3,train_loader_4, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_ndcg2.pkl')))
        _, _, _, _, test_ndcg2, _ = test_process(model, train_loader_3,train_loader_4, feed_data, args.cuda, topK, mode='test')
        model.load_state_dict(torch.load(os.path.join(log, 'best_mrr2.pkl')))
        _, _, _, _, _, test_mrr2 = test_process(model, train_loader_3,train_loader_4, feed_data, args.cuda, topK, mode='test')
        print('Test TopK:{} ---> movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
              .format(topK, test_hr1, test_ndcg1, test_mrr1, test_hr2, test_ndcg2, test_mrr2))
        with open(log + '/tmp.txt', 'a') as f:
            f.write('Test TopK:{} ---> movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}\n'
                    .format(topK, test_hr1, test_ndcg1, test_mrr1, test_hr2, test_ndcg2, test_mrr2))



if __name__ == "__main__":
    print(args)
    main()

