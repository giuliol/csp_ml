comando usato per rinominare in batch.

ls | cat -n | while read n f; do mv "$f" "$n.extension"; done 

