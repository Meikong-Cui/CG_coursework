# Computer Graphics coursework

This coursework is a c++ renderer, which support not only rasterising and ray-tracing, but also some advanced features like soft shadow, Goarud shading and Phong shading. The renderer will automatically rotate the camera around the origin point and use cornell-box as default model.

The video shows features of render: 
reflection			        0:00 - 0:07
softshodow 		          0:07 - 0:13
raytracing texture		  0:13 - 0:24
proximity & incident	  0:24 - 0:30
specular			          0:30 - 0:36
rasterise & wireframe	  0:36 - 0:55
rasterise texture		    0:55 - 1:05
rayTracing Phong		    1:05 - 1:10

There are two files containing the code, the difference between them is the .obj file. You can run all the features of cornell-box in folder 'cw' and check Phong shading in the 'sphere' folder. Both of them can be compiled by make.

some key to cotrol the render engine: 
'w, a, s, d, q, e'		-	camera ratation
'up, down, left, right, ., /'	-	camera translation
'i, j, k, l, [, ]'		-	light translation
'1, 2, 3'			-	rasterise, ray tracing, wireframe
'4 ,5, 6, 7'			-	proximity, incident, specular, phong
'9'			-	softshadow
'u'			-	random stroked triangle
'f'			-	random filled triangle
't'			-	swith of texture
'm, r'			-	render one step of model in rasterise/raytracing
'h'			-	swith of the auto ratation
'0'			-	look at (0, 0, 0)

After compile, the model will rotate automatically. You may want to change line 743 'bool draw = true' to 'bool draw = false' or press 'h' and use key 'm' and 'r' to render the model frame by frame. 'sphere' only have key control on camera and light.
