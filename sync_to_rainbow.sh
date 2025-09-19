#!/bin/zsh

command="rsync --delete -avz --human-readable --progress \
    ../real --exclude-from=.rsyncignore --delete \
    rainbow:/home/nvidia/workbench"

# echo "\033[93mThe following command will be executed:\033[0m"
# echo "$command"

# read -p "Continue? (y/n): " response

# if [ "$response" = "y" ]; then
#   $command
# else
#   echo "Command cancelled."
# fi

$command