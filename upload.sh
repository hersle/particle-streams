#!/bin/sh

trap exit INT # prevent write of incomplete video upon SIGINT (kill whole script instead)

tmpdestfile="/tmp/tmpanim.mp4" # temporary location (so final video is quickly moved at the end)

for fmkv in $(find -name "*.mkv"); do
	fmp4="$(echo $fmkv | rev | cut -c 4- | rev)mp4" # .mkv -> .mp4
	if [ -f $fmp4 ]; then
		echo "Already converted $fmkv"
	else
		echo "Converting $fmkv ..."
		ffmpeg -y -v warning -i $fmkv -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p $tmpdestfile # add -stats and -hide_banner for progress
		mv "$tmpdestfile" "$fmp4"
	fi

	echo "Removing $fmkv ..."
	rm "$fmkv"
done

# rsync -vtr . --include="*/" --include="*.mp4" --exclude="*" hermasl@login.stud.ntnu.no:public_html/particle_collisions # upload to web
