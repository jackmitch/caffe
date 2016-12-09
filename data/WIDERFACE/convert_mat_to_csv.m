m = load('C:\Users\JLeigh\Downloads\wider_face_split\v1\wider_face_val.mat');

fid = fopen('C:\Users\JLeigh\Downloads\wider_face_split\v1\wider_face_val.csv', 'w') ;

[num_scenes,s] = size(m.file_list);

fprintf(fid, 'filename,left,top,width,height\n');

for i=1:num_scenes
    scene_boxes = m.face_bbx_list{i};
    [num_imgs_in_scene, s] = size(scene_boxes);
    scene_filenames = m.file_list{i};
    for j=1:num_imgs_in_scene
        image_filename = strcat(m.event_list{i}, '/', scene_filenames{j});
        image_boxes = scene_boxes{j};
        [num_boxes_in_img, s] = size(image_boxes);
        for n=1:num_boxes_in_img
            box = image_boxes(n,:);
            fprintf(fid, '%s,%f,%f,%f,%f\n', image_filename, ...
                    box(1,1), box(1,2), box(1,3), box(1,4));
        end
    end
end

fclose(fid) ;