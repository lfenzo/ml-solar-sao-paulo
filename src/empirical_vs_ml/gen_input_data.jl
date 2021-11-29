"""
Obtemos valores da radiação solar diária (o que é trapaça)
para o cálculo do modelo empírico.
"""

using Dates
using CSV
using DataFrames

function daily_gsr(data) :: DataFrame

  gsr_sum = DataFrame(doy = Int64[],
                      year = Int64[],
                      daily_gsr = Float64[])

  for daily in filter(g -> size(g, 1) == 11, groupby(data, [:year, :doy]))
  
    doy  = daily[begin, :doy] 
    year = daily[begin, :year]

    info = (doy, year, sum(daily[:, :radiacao_global])) 

    push!(gsr_sum, info)
  end

  return gsr_sum
end

function main()

  DATA_DIR_ROOT = "../data/estacoes_processadas/"
  DATETIME_FORMAT = "yyyy-mm-dd HH:MM:SS"
  DST_DIR_ROOT = "./input_data/"

  datafile = "A509.csv"
  station_id = split(datafile, ".")[begin]
  
  station_metadata = CSV.read( "../data/station_status.csv", DataFrame )

  #
  # cria a parta de destino dos arquivos .csv criados aqui (caso ela não exista)
  if !isdir(DST_DIR_ROOT)
    mkdir(DST_DIR_ROOT)
  end

  # encontrando todos os arquivos com extensão .csv
  all_datafiles = filter(v -> occursin("csv", split(v, ".")[2]), readdir(DATA_DIR_ROOT))

  # selecionando apenas os
  #train_stations = filter(row -> row.status == "train", station_metadata)[:, :station]
  all_stations = map(s -> split(s, ".")[begin], all_datafiles)
 

  for station_id in all_stations

    println("Processando $station_id")

    data = CSV.read( joinpath(DATA_DIR_ROOT, "$station_id.csv"), DataFrame )

    transform!(data, :datetime => ByRow( r -> DateTime(split(r, "+")[begin], DATETIME_FORMAT) ) => :datetime)

    transform!(data, :datetime => ByRow(dayofyear) => :doy)
    transform!(data, :datetime => ByRow(year) => :year)
    transform!(data, :datetime => ByRow(hour) => :hour)

    select!(data, [:doy, :hour, :year, :radiacao_global, :rad_prox_hora])

    data = data[ 2019 .<= data[:, :year] .<= 2021, :]

    daily_gsr_df = daily_gsr(data)

    filtered_valid = outerjoin(data, daily_gsr_df, on = [:doy, :year])

    dropmissing!(filtered_valid)
    
    lat = station_metadata[ station_metadata[:, :station] .== station_id, :lat]
    lon = station_metadata[ station_metadata[:, :station] .== station_id, :lon]

    if isempty(filtered_valid) || isempty(lat) || isempty(lon)
      continue
    end

    insertcols!(filtered_valid, 1, :lat => repeat(lat, size(filtered_valid, 1)))
    insertcols!(filtered_valid, 1, :lon => repeat(lon, size(filtered_valid, 1)))

    CSV.write("./input_data/$station_id.csv", filtered_valid)

  end

end


main()
