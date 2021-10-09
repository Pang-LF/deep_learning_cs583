function weight = randomlearningf(weight,loss)
    weightnew = -5 + (5+5)*rand;
    prevalue = weight * dlX
    losstry = crossentropy(prevalue, dlY)
    if losstry < loss
        loss = losstry
        weight = weightnew
    end 
end
