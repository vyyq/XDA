#!/bin/bash
rm -rf checkpoints
mkdir -p checkpoints/pretrain_all
mkdir -p checkpoints/finetune_msvs_funcbound_64


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k1en42j749BEtr5-AFEjC9cvFy9Y5zDI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1k1en42j749BEtr5-AFEjC9cvFy9Y5zDI" -O ./checkpoints/pretrain_all/checkpoint_best.pt && rm -rf /tmp/cookies.txt


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1103Hq2ZShlF-4qRPudtjDru5fBqAckds' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1103Hq2ZShlF-4qRPudtjDru5fBqAckds" -O ./checkpoints/finetune_msvs_funcbound_64/checkpoint_best.pt && rm -rf /tmp/cookies.txt
