import torch
import numpy as np

def flatten(t):
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t

def verify_setup():
    print(torch.__version__)
    torch.cuda.is_available()
    print(torch.version.cuda)

def main():
    verify_setup()
    dd = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
    t = torch.tensor(dd)
    type(t)
    t.shape
    print (t.dtype)
    print (t.device)
    print (t.layout)

    data = np.array([1, 2, 3])
    type (data)

    ### class constructor
    t1 = torch.Tensor(data)   

    ### factory functions.
    t2 = torch.tensor(data)
    t3 = torch.as_tensor(data)
    t4 = torch.from_numpy(data)

    print(t1.dtype)
    print(t2.dtype)
    print(t3.dtype)
    print(t4.dtype)

    t5 = torch.tensor(np.array([1, 2, 3]))    
    t6 = torch.tensor(np.array([1., 2., 3.]))
    t7 = torch.tensor(np.array([1, 2, 3]),dtype = torch.float64)    

    data = np.array([1, 2, 3])

    t1 = torch.Tensor(data)   
    t2 = torch.tensor(data)
    t3 = torch.as_tensor(data)
    t4 = torch.from_numpy(data)

    data[0] = 0
    data[1] = 0
    data[2] = 0

    print (torch.eye(2))
    print (torch.zeros(2,2))
    print (torch.ones(2,2))
    print (torch.rand(2,2))

    ### rank is 2: takes two index to acces any point.
    ### axis is 2: length of first axis 3, second axis 4.
    t = torch.tensor([[1,1,1,1],
                    [2,2,2,2],
                    [3,3,3,3]], dtype=torch.float32)

    print(t.size())
    print(t.shape)
    print(torch.tensor(t.shape).prod())
    
    print(t.numel())
    ### t = t.cude()
    print(t.reshape(3,4))
    print(t.reshape(4,3))
    print(t.reshape(6,2))
    print(t.reshape(2,6))
    print(t.reshape(2,2,3))
    print(t.reshape(1,12))
    print(t.reshape(1,12).shape)
    print(t.reshape(1,12).squeeze())
    print(t.reshape(1,12).squeeze().shape)
    print(t.reshape(1,12).squeeze().unsqueeze(dim=0))
    print(t.reshape(1,12).squeeze().unsqueeze(dim=0).shape)

    t1 = torch.tensor([[1,2],[3,4]])
    t2 = torch.tensor([[5,6],[7,8]])
    print(torch.cat((t1,t2),dim=1))

    t1 = torch.tensor([[1,1,1,1],
                    [1,1,1,1],
                    [1,1,1,1],
                    [1,1,1,1]])

    t2 = torch.tensor([[2,2,2,2],
                    [2,2,2,2],
                    [2,2,2,2],
                    [2,2,2,2]])

    t3 = torch.tensor([[3,3,3,3],
                     [3,3,3,3],
                     [3,3,3,3],
                     [3,3,3,3]])

    t = torch.stack((t1,t2,t3))
    print(t)
    t.shape
    t = t.reshape(3,1,4,4,)
    print(t)
    print(t[0])
    print(t[0][0])
    print(t[0][0][0])
    print(t[0][0][0][0])

    # t.reshape(1,-1)[0]
    # t.reshape(-1)
    # t.view(t.numel())
    # t.flatten()
    print(t.flatten(start_dim=1).shape)
    print(t.flatten(start_dim=1))

    t1 = torch.tensor([[1,2], [3,4]], dtype=torch.float32)    
    t2 = torch.tensor([[5,6], [7,8]], dtype=torch.float32)
    print(t1)
    print(t2)
    print(t1 + t2)

    print (t1[0])
    print (t1[0][0])
    print (t2[0][0])

    print( t1 + 2)
    print( t1.add(2))

    print( t1 - 2)
    print( t1.sub(2))

    print( t1 * 2)
    print( t1.mul(2))

    print( t1 / 2)
    print( t1.div(2))

    print (np.broadcast_to(2, t1.shape))
    t3 = t1 + torch.tensor(np.broadcast_to(2, t1.shape), dtype=torch.float32)

    t1 = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    t2 = torch.tensor([2, 4], dtype=torch.float32)
    t1.shape
    t2.shape

    print (t1 + t2)

    t = torch.tensor([[0, 5, 7],
                     [6, 0, 7],
                     [0, 8, 0]], dtype=torch.float32)
    print (t)
    print ( t.eq(0))
    print ( t.ge(0))
    print ( t.gt(0))
    print ( t.lt(0))
    print ( t.le(7))

    t = torch.tensor([1, 2, 3]) < torch.tensor([3, 1, 2])
    print(t)

    t = torch.tensor([[0, 5, 7],
                     [6, 0, 7],
                     [0, 8, 0]], dtype=torch.float32)
    print ( t.abs())    
    print ( t.sqrt())
    print ( t.neg())

    t = torch.tensor([[0, 1, 0],
                      [2, 0, 2],
                      [0, 3, 0]], dtype=torch.float32)

    print( t.sum() )
    print( t.numel() )
    print( t.sum().numel() )
    print( t.sum().numel() < t.numel() )

    print( t.sum() )
    print( t.prod() )
    print( t.mean() )
    print( t.std() )

    t = torch.tensor([[1,1,1,1],
                      [2,2,2,2],
                      [3,3,3,3]], dtype=torch.float32)
    
    print ( t.sum(dim=0) )
    print ( t.sum(dim=1) )

    print( t[0] )
    print( t[1] )
    print( t[2] )
    print(t[0] + t[1] + t[2])

    print( t[0].sum() )
    print( t[1].sum() )
    print( t[2].sum() )
    
    t = torch.tensor([[1,0,0,2],
                      [0,3,3,0],
                      [4,0,0,5]], dtype=torch.float32)

    print(t.max())
    print(t.argmax())

    print ( t.flatten())
    print ( t.max(dim=0))
    print ( t.argmax(dim=0))
    print ( t.max(dim=1))
    print ( t.argmax(dim=1))

    t = torch.tensor([[1,2,3],
                      [4,5,6],
                      [7,8,9]],dtype=torch.float32)
    print ( t.mean())
    print (t.mean().item())
    print (t.mean(dim=0).tolist())
    print (t.mean(dim=0).numpy())

    print("hello world")


if __name__ == "__main__":
    main()
    print("Got this far.")
