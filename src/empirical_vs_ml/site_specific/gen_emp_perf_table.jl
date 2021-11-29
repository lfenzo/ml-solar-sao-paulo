include("../empirical.jl")

using MLJ
using CSV
using DataFrames
using Statistics

using ..EmpiricalModels

"""
Implementa o cáculo do Mean Bias Error
"""
function mbe(preds::AbstractVector, real::AbstractVector)
    return sum((real .- preds)) / length(preds)
end



"""
Implementa o cáculo do R2 Score 
"""
function r2(preds::AbstractVector, real::AbstractVector)

    real_mean = mean(real)

    var_Y_hat = sum((preds .- real_mean).^2)
    var_Y_bar = sum((real .- real_mean).^2)

    return (var_Y_bar - var_Y_hat) / var_Y_bar
end


function main()

    if !isdir("./empirical_preds_julia/")
        mkdir("./empirical_preds_julia/")
    end
    
    emp_perf_compare = DataFrame(station = String[],
                                 rmse = Float64[],
                                 mae = Float64[],
                                 mbe = Float64[],
                                 r2 = Float64[])
       
    for file in readdir("./input_data")

        input_data = CSV.read("./input_data/$file", DataFrame)

        if isempty(input_data)
            continue
        end

        println("processando $file...")

        this_site_preds = DataFrame(doy = Int64[],
                                    hour = Int64[],
                                    year = Int64[],
                                    emp_pred = Float64[],
                                    real = Float64[])


        lat = input_data[begin, :lat]
        lon = input_data[begin, :lon]

        # gera as previsões e as guara em conjuntos de previsoes
        for row in eachrow(input_data)

            localtime = row[:hour] - 3
            standard_meridian_time = -3
            doy = row[:doy]
            total_daily = row[:daily_gsr]

            emp_pred = EmpiricalModels.predict(lat, lon, localtime, standard_meridian_time, doy, total_daily)

            info = (doy, localtime, row[:year], emp_pred, row[:radiacao_global])
            push!(this_site_preds, info)

        end

        CSV.write("./empirical_preds_julia/$file" , this_site_preds)

        pred = this_site_preds[:, :emp_pred]
        real = this_site_preds[:, :real]

        perf_info = (split(file, ".")[begin],
                     rmse(pred, real),
                     mae(pred, real),
                     mbe(pred, real),
                     r2(pred, real) )


        push!(emp_perf_compare, perf_info)
    
    end

    CSV.write("./empirical_performance_julia.csv" , emp_perf_compare)

end

main()

    
