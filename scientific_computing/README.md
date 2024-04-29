=====================================================================

===================xyz-sources=======================================

=====================================================================

git clone git@code.xyz.com:xyz/xyz-sources.git

cd xyz-sources

git branch --list

git checkout -t origin/v5.5

 

O:\dev\xyzexe\xyz_5.5\xyz 5.5 (5554)\xyz 5.5 (5554)\bin

 

ls -R -l grep run: > hello.txt

 

=====================================================================

======================sphinx=========================================

=====================================================================

sphinx-quickstart

 

import os

import sys

sys.path.insert(0, os.path.abspath('..'))

sys.path.append('C:/Users/farad/mytool/code_package')

 

make clean

 

make html

 

sphinx-apidoc -o "C:/Users/farad/mytool/sphinx/rst/" "C:/Users/farad/mytool/code_package/"

sphinx-apidoc -o . ../code_package

sphinx-apidoc -o . "C:/Users/farad/mytool/code_package/"

 

sphinx-build -b html "C:/Users/farad/mytool/sphinx/rst/" "C:/Users/farad/mytool/sphinx/html/"

 

=====================================================================

pip --trusted-host XYZ.XYZ.int install

git clone https://github.com/xyz/XYZ.git

jupyter notebook

 

git submodule update --init

git branch -a

 

git remote show origin

git push -u origin <branch name>

 

======================================================

============Daily Command Samples===================

======================================================

ls -R .

ls -lt

ls -halt

ls -li

 

find . -print

find . -print -ls

find . -type f -iname '*.tpl'

find . -type f -iname '*.ctl' -print0 | xargs -I {} -0 grep -H --color 'xyz_xyz' "{}"

 

grep -r xy_xyz --include=\*.dpl

grep -l version *

grep -r version *

grep -n version *

grep -ri ldap

grep -ri ldap | grep xml

 

=======================

htop               //memory usage.

tp                 //memory usage per process.

vmstat -s

cat /proc/meminfo

free -m

ln -s target src   //soft link - have different inode.

ln target src      //hard link - have the same inode

 

sudo parted -l

sudo mount /dev/sdb1 /my_usb/   

 

cat /etc/fstab                         //UUID = Universal Unique ID.

du -a

ps aux

gdb -v

g++ -v

make -v

 

git tag v.0.0.1                                                // Mark commit with a tag name.

git tag --list                                                    // view tags

git push --tags                                               // push tags.

git checkout -b dev v.0.0.1           // check out v1

git tag -a v0.0.1 -m "Version 0.0.0 of ..." //Create Annotated Tags.

git show v.0.0.1

git push origin -d v.0.0.1

git tag v.0.0.1 83606e768e

 

git pull

git status

git fetch

git rebase origin/master

git branch -b branchName   

git branch branchName

git checkout branchName

 

git commit -am "comments".

git log

git checkout -- file_name.py

git blame xyz_xyz.py | less

git cherry-pick 000000bbbbbbbffffffaaaaaaabcdabcd

git branch -r | grep sub               //gives the name of the branches including sub.

git reflog

 

git reset --hard temp

git fetch

git checkout -b Release_xyz.xyz origin/Release_xyz.xyz

git checkout -b Release_xyz.xyz origin/Release_xyz.xyz

 

# steps for sqashing:

# rebase your branch to top of the origin HEAD

 

# push your changes to the origin.

git fetch

git rebase -i HEAD~13

git rebase origin/Release_xyz

git push -f origin HEAD

 

git diff Release_1.8 Release_1.9 filename.py

 

git remote set-url origin git@code.xxx.com:xyz/dev_env.git

git remote set-url origin git@code.xxx.com:xyz/simulator.git

git remote -v

git pull

 

 

=====Create a new branch=====/home/devel/devenv/some_testing====

cd devenv/

git status                   //status of the working directory and the staging area.

git fetch                    //Downloads commits etc.. this is what you do when you want to see what others have done.

gitk

gitk --all

git checkout -b some_branch_name   //allows you to navigate between branches.

 

git add mqm.py                                       //tell the depository that you consider making some changes.

git status

git commit -am "init"                                //Commit set a message about the changes you were done

 

git status

git push origin some_branch_name     //upload local repository changes to a remote rep.

 

git branch -r | grep helloworld                          //reference to a commit.

gitk origin/dev_branch

 

git push -f origin dev_branch_name

git checkout Release_xyz

 

git merge --ff-only dev_branch_name

git push origin Release_xyz

 

git branch -d dev_branch_name

git push origin --delete dev_branch_name

 

 

===============================================================

sudo yum install git-gui

Makefile

g++ main.cpp file1.cpp

g++ main.o file1.o -o output

g++ -c main.cpp

 

==============================================================================

netstat -tulpn | grep LISTEN

lsof -i -P -n | grep LISTEN

ss -tulpn | grep LISTEN

lsof -i :1414

ls -l /proc/26454/exe

 

 

================================================

git log -p filename

git log --oneline -3 filename

git log -3 filename

git log --since='2021-04-01' filename

git log --oneline #gives sha-keys

git log 2d081d7..4e9104e #firs sha-key is bottom and second is top sha-key.

git diff-tree --no-commit-id --name-only -r sha-keys

gitk filename
