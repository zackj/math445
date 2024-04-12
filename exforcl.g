alphabet:=[0..25];
a:=[ 1, 3, 5,  7,  9,  11,  15, 17, 19, 21,  23, 25 ];
b:=[0..25];
affglob:=function ( r, a, b, n )
    return List( r, 
           x-> ( a*x+b) mod  n );
end;
sol:=0;
for z in a do
for t in b do
  if  alphabet = affglob(affglob(alphabet,z,t,26),z,t,26) then 
    Print("a is ",z," b is ",t,"\n");
    sol:=sol+1;
  fi;
od;
od;
Print("the number of solutions is ",sol,"\n");
